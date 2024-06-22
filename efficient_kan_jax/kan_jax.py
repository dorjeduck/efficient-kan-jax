import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np

class KANLinear(nn.Module):
    """
    Linear layer for KAN model using JAX/Flax.
    """

    in_features: int
    out_features: int
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    enable_standalone_scale_spline: bool = True
    base_activation: nn.Module = nn.silu
    grid_eps: float = 0.02
    grid_range: list = (-1, 1)

    def setup(self):
        # Compute grid based on grid_range and grid_size
        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid = jnp.arange(-self.spline_order, self.grid_size + self.spline_order + 1) * h + self.grid_range[0]
        grid = jnp.expand_dims(grid, axis=0).repeat(self.in_features, axis=0)
        self.grid = self.param("grid", lambda rng: grid)  # Register grid as a parameter

        # Initialize learnable parameters
        self.base_weight = self.param("base_weight", nn.initializers.lecun_normal(), (self.out_features, self.in_features))
        self.spline_weight = self.param("spline_weight", nn.initializers.lecun_normal(), (self.out_features, self.in_features, self.grid_size + self.spline_order))
        
        if self.enable_standalone_scale_spline:
            self.spline_scaler = self.param("spline_scaler", nn.initializers.ones, (self.out_features, self.in_features))

    def b_splines(self, x):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (jax.ndarray): Input tensor of shape (batch_size, in_features).

        Returns:
            jax.ndarray: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features

        grid = self.grid
        x = jnp.expand_dims(x, -1)
        bases = jnp.where((x >= grid[:, :-1]) & (x < grid[:, 1:]), 1.0, 0.0)
        
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] + 
                     (grid[:, k+1:] - x) / (grid[:, k+1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])

        return bases

    def curve2coeff(self, x, y):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (jax.ndarray): Input tensor of shape (batch_size, in_features).
            y (jax.ndarray): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            jax.ndarray: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features
        assert y.shape == (x.shape[0], self.in_features, self.out_features)

        A = jnp.swapaxes(self.b_splines(x), 0, 1)
        B = jnp.swapaxes(y, 0, 1)
        
        solution = jnp.linalg.lstsq(A, B, rcond=None)[0]
        result = jnp.swapaxes(solution, 1, 2)
        
        return result

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler[..., None] if self.enable_standalone_scale_spline else 1.0)

    def __call__(self, x):
        """
        Forward pass of the linear layer.

        Args:
            x (jax.ndarray): Input tensor of shape (batch_size, in_features).

        Returns:
            jax.ndarray: Output tensor of shape (batch_size, out_features).
        """
        assert x.shape[-1] == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = self.base_activation(x) @ self.base_weight.T
        spline_output = (self.b_splines(x).reshape(x.shape[0], -1) @ self.scaled_spline_weight.reshape(self.out_features, -1).T)
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    def update_grid(self, x, margin=0.01):
        """
        Update the grid used for B-splines based on the input data.

        Args:
            x (jax.ndarray): Input tensor of shape (batch_size, in_features).
            margin (float, optional): Margin to adjust grid points. Defaults to 0.01.
        """
        assert x.ndim == 2 and x.shape[1] == self.in_features
        batch = x.shape[0]

        splines = self.b_splines(x).transpose(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.transpose(1, 2, 0)
        unreduced_spline_output = jnp.einsum('ijk,ikl->ijl', splines, orig_coeff).transpose(1, 0, 2)

        x_sorted = jnp.sort(x, axis=0)
        grid_adaptive = x_sorted[jnp.linspace(0, batch - 1, self.grid_size + 1, dtype=int)]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (jnp.arange(self.grid_size + 1)[:, None] * uniform_step + x_sorted[0] - margin)
        
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = jnp.concatenate([grid[:1] - uniform_step * jnp.arange(self.spline_order, 0, -1)[:, None], grid, 
                                grid[-1:] + uniform_step * jnp.arange(1, self.spline_order + 1)[:, None]], axis=0)
        
        self.grid = self.grid.at[:].set(grid.T)
        self.spline_weight = self.spline_weight.at[:].set(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This function computes the regularization loss using L1 regularization on spline weights.

        Args:
            regularize_activation (float, optional): Weight for activation regularization. Defaults to 1.0.
            regularize_entropy (float, optional): Weight for entropy regularization. Defaults to 1.0.

        Returns:
            jax.ndarray: Regularization loss value.
        """
        l1_fake = jnp.mean(jnp.abs(self.spline_weight), axis=-1)
        regularization_loss_activation = jnp.sum(l1_fake)
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -jnp.sum(p * jnp.log(p))
        return (regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy)

class KAN(nn.Module):
    """
    Multi-layered KAN model using JAX/Flax.
    """

    layers_hidden: list
    grid_size: int = 5
    spline_order: int = 3
    scale_noise: float = 0.1
    scale_base: float = 1.0
    scale_spline: float = 1.0
    base_activation: nn.Module = nn.silu
    grid_eps: float = 0.02
    grid_range: list = (-1, 1)
    
    def setup(self):
        # Initialize layers based on hidden sizes
        self.layers = [KANLinear(in_features, out_features, 
                                 grid_size=self.grid_size, 
                                 spline_order=self.spline_order, 
                                 scale_noise=self.scale_noise, 
                                 scale_base=self.scale_base, 
                                 scale_spline=self.scale_spline, 
                                 base_activation=self.base_activation, 
                                 grid_eps=self.grid_eps, 
                                 grid_range=self.grid_range)
                       for in_features, out_features in zip(self.layers_hidden, self.layers_hidden[1:])]

    def __call__(self, x, update_grid=False):
        """
        Forward pass of the KAN model.

        Args:
            x (jax.ndarray): Input tensor of shape (batch_size, in_features).
            update_grid (bool, optional): Whether to update grids in KANLinear layers. Defaults to False.

        Returns:
            jax.ndarray: Output tensor of shape (batch_size, out_features).
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss across all layers.

        Args:
            regularize_activation (float, optional): Weight for activation regularization. Defaults to 1.0.
            regularize_entropy (float, optional): Weight for entropy regularization. Defaults to 1.0.

        Returns:
            jax.ndarray: Regularization loss value.
        """
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)
