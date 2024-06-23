import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from flax.training.common_utils import onehot
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

class SplineLinear(nn.Module):
    in_features: int
    out_features: int
    init_scale: float = 0.1

    def setup(self):
        self.weight = self.param('weight', nn.initializers.truncated_normal(stddev=self.init_scale), (self.out_features, self.in_features))

    def __call__(self, x):
        return jnp.dot(x, self.weight.T)


class RadialBasisFunction(nn.Module):
    grid_min: float = -2.0
    grid_max: float = 2.0
    num_grids: int = 8
    denominator: float = None

    def setup(self):
        grid = jnp.linspace(self.grid_min, self.grid_max, self.num_grids)
        self.grid = self.variable('params', 'grid', lambda: grid)
        denominator_value = self.denominator if self.denominator is not None else (self.grid_max - self.grid_min) / (self.num_grids - 1)
        self.denom = self.variable('params', 'denominator', lambda: denominator_value)

    def __call__(self, x):
        return jnp.exp(-((x[..., None] - self.grid.value) / self.denom.value) ** 2)


class FastKANLayer(nn.Module):
    input_dim: int
    output_dim: int
    grid_min: float = -2.0
    grid_max: float = 2.0
    num_grids: int = 8
    use_base_update: bool = True
    use_layernorm: bool = True
    base_activation: nn.Module = nn.silu
    spline_weight_init_scale: float = 0.1

    def setup(self):
        if self.use_layernorm:
            assert self.input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm()
        else:
            self.layernorm = None

        self.rbf = RadialBasisFunction(grid_min=self.grid_min, grid_max=self.grid_max, num_grids=self.num_grids)
        self.spline_linear = SplineLinear(self.input_dim * self.num_grids, self.output_dim, init_scale=self.spline_weight_init_scale)

        if self.use_base_update:
            self.base_linear = nn.Dense(self.output_dim)

    def __call__(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            x = self.layernorm(x)
        spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.reshape(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

class FastKAN(nn.Module):
    layers_hidden: list
    grid_min: float = -2.0
    grid_max: float = 2.0
    num_grids: int = 8
    use_base_update: bool = True
    base_activation: nn.Module = nn.silu
    spline_weight_init_scale: float = 0.1

    def setup(self):
        self.layers = [
            FastKANLayer(
                input_dim=in_dim,
                output_dim=out_dim,
                grid_min=self.grid_min,
                grid_max=self.grid_max,
                num_grids=self.num_grids,
                use_base_update=self.use_base_update,
                base_activation=self.base_activation,
                spline_weight_init_scale=self.spline_weight_init_scale
            ) for in_dim, out_dim in zip(self.layers_hidden[:-1], self.layers_hidden[1:])
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionWithFastKANTransform(nn.Module):
    q_dim: int
    k_dim: int
    v_dim: int
    head_dim: int
    num_heads: int
    gating: bool = True

    def setup(self):
        total_dim = self.head_dim * self.num_heads
        self.linear_q = FastKANLayer(self.q_dim, total_dim)
        self.linear_k = FastKANLayer(self.k_dim, total_dim)
        self.linear_v = FastKANLayer(self.v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, self.q_dim)

        if self.gating:
            self.linear_g = FastKANLayer(self.q_dim, total_dim)
        else:
            self.linear_g = None

        self.norm = self.head_dim**-0.5

    def __call__(self, q, k, v, bias=None):
        wq = self.linear_q(q).reshape(*q.shape[:-1], self.num_heads, -1) * self.norm
        wk = self.linear_k(k).reshape(*k.shape[:-2], k.shape[-2], self.num_heads, -1)
        att = jnp.einsum('...qhd,...khd->...qkh', wq, wk).softmax(-2)
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v).reshape(*v.shape[:-2], v.shape[-2], self.num_heads, -1)
        o = jnp.einsum('...qkh,...khd->...qhd', att, wv)
        o = o.reshape(*o.shape[:-2], -1)

        if self.linear_g is not None:
            g = self.linear_g(q)
            o = nn.sigmoid(g) * o

        o = self.linear_o(o)
        return o