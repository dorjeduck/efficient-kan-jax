import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm

def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

@jax.jit
def eval_step(state, images, labels):
    logits = state.apply_fn({'params': state.params}, images)
    loss = cross_entropy_loss(logits, labels)
    return loss, logits

class Trainer:
    def __init__(self, model_cls, layers_hidden, learning_rate=1e-3, weight_decay=1e-4, epochs=10, batch_size=64, data_loader=None):
        self.model = model_cls(layers_hidden)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

        key = jax.random.PRNGKey(0)
        self.params = self.model.init(key, jnp.ones((1, 28 * 28)))['params']

        self.tx = optax.adamw(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=self.tx)

        if data_loader:
            self.trainloader, self.valloader = data_loader(self.batch_size)
        else:
            self.trainloader, self.valloader = None, None

    def train_and_evaluate(self):
        for epoch in range(self.epochs):
            train_loss = 0
            train_accuracy = 0
            update_frequency = 10

            with tqdm(self.trainloader, desc=f"Epoch {epoch + 1} [Training]") as pbar:
                cumulative_loss = 0.0
                cumulative_accuracy = 0.0
                num_batches = len(self.trainloader)

                for i, (images, labels) in enumerate(pbar):
                    images = jnp.array(images.view(-1, 28 * 28).numpy())
                    labels = jnp.array(labels.numpy())
                    self.state, loss, logits = train_step(self.state, images, labels)
                    train_loss += loss
                    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
                    train_accuracy += accuracy

                    cumulative_loss += loss.item()
                    cumulative_accuracy += accuracy.item()

                    if i % update_frequency == 0 or i + 1 == num_batches:
                        avg_loss = cumulative_loss / update_frequency
                        avg_accuracy = cumulative_accuracy / update_frequency

                        pbar.set_postfix(loss=avg_loss, accuracy=avg_accuracy, lr=self.learning_rate)

                        cumulative_loss = 0.0
                        cumulative_accuracy = 0.0

            train_loss /= len(self.trainloader)
            train_accuracy /= len(self.trainloader)
            print(f"Epoch {epoch + 1} [Training]: Loss: {train_loss}, Accuracy: {train_accuracy}\n")

            val_loss = 0
            val_accuracy = 0
            with tqdm(self.valloader, desc=f"Epoch {epoch + 1} [Validation]") as pbar:
                cumulative_loss = 0.0
                cumulative_accuracy = 0.0
                num_batches = len(self.valloader)
                for i, (images, labels) in enumerate(pbar):
                    images = jnp.array(images.view(-1, 28 * 28).numpy())
                    labels = jnp.array(labels.numpy())
                    loss, logits = eval_step(self.state, images, labels)
                    val_loss += loss
                    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
                    val_accuracy += accuracy

                    cumulative_loss += loss.item()
                    cumulative_accuracy += accuracy.item()

                    if i % update_frequency == 0 or i + 1 == num_batches:
                        avg_loss = cumulative_loss / update_frequency
                        avg_accuracy = cumulative_accuracy / update_frequency

                        pbar.set_postfix(loss=avg_loss, accuracy=avg_accuracy)

                        cumulative_loss = 0.0
                        cumulative_accuracy = 0.0

            val_loss /= len(self.valloader)
            val_accuracy /= len(self.valloader)
            print(f"Epoch {epoch + 1} [Validation]: Loss: {val_loss}, Val Accuracy: {val_accuracy}\n")

            self.learning_rate *= 0.8
            self.tx = optax.adamw(learning_rate=self.learning_rate, weight_decay=self.weight_decay)
            self.state = train_state.TrainState.create(
                apply_fn=self.model.apply, params=self.state.params, tx=self.tx)
