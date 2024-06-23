import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from efficient_kan_jax import KAN  

# Load MNIST data with transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root="./data", train=True,
                          download=True, transform=transform)
valset = datasets.MNIST(root="./data", train=False,
                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Define the model
model = KAN([28 * 28, 64, 10])
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 28 * 28)))['params']

# Define optimizer
learning_rate = 1e-3
tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params, tx=tx)

# Cross entropy loss function
def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

# Training step
@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    # Compute gradients and loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)

    # Apply gradients to update parameters
    state = state.apply_gradients(grads=grads)

    return state, loss, logits

# Evaluation step
@jax.jit
def eval_step(params, images, labels):
    logits = model.apply({'params': params}, images)
    loss = cross_entropy_loss(logits, labels)
    return loss, logits

# Training and evaluation loop
for epoch in range(10):
    # Train
    train_loss = 0
    train_accuracy = 0

    update_frequency = 10

    with tqdm(trainloader, desc=f"Epoch {epoch + 1} [Training]") as pbar:
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0
        num_batches = len(trainloader)

        for i, (images, labels) in enumerate(pbar):
            images = jnp.array(images.view(-1, 28 * 28).numpy())
            labels = jnp.array(labels.numpy())
            state, loss, logits = train_step(state, images, labels)
            train_loss += loss
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            train_accuracy += accuracy

            cumulative_loss += loss.item()
            cumulative_accuracy += accuracy.item()

            if i % update_frequency == 0 or i + 1 == num_batches:
                avg_loss = cumulative_loss / update_frequency
                avg_accuracy = cumulative_accuracy / update_frequency

                pbar.set_postfix(loss=avg_loss, accuracy=avg_accuracy, lr=learning_rate)

                # Reset cumulative values
                cumulative_loss = 0.0
                cumulative_accuracy = 0.0

    train_loss /= len(trainloader)
    train_accuracy /= len(trainloader)
    print(f"Epoch {epoch + 1} [Training]: Loss: {train_loss}, Accuracy: {train_accuracy}\n")

    # Validation
    val_loss = 0
    val_accuracy = 0
    with tqdm(valloader, desc=f"Epoch {epoch + 1} [Validation]") as pbar:
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0
        num_batches = len(valloader)
        for i, (images, labels) in enumerate(pbar):
            images = jnp.array(images.view(-1, 28 * 28).numpy())
            labels = jnp.array(labels.numpy())
            loss, logits = eval_step(state.params, images, labels)
            val_loss += loss
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
            val_accuracy += accuracy

            cumulative_loss += loss.item()
            cumulative_accuracy += accuracy.item()

            if i % update_frequency == 0 or i + 1 == num_batches:
                avg_loss = cumulative_loss / update_frequency
                avg_accuracy = cumulative_accuracy / update_frequency

                pbar.set_postfix(loss=avg_loss, accuracy=avg_accuracy)

                # Reset cumulative values
                cumulative_loss = 0.0
                cumulative_accuracy = 0.0

    val_loss /= len(valloader)
    val_accuracy /= len(valloader)
    print(f"Epoch {epoch + 1} [Validation]: Loss: {val_loss}, Val Accuracy: {val_accuracy}\n")

    # Update learning rate
    learning_rate *= 0.8
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=state.params, tx=tx)
