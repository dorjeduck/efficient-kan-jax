from efficient_kan_jax import KAN as EfficientKAN_JAX
from fastkan_jax import  FastKAN as FastKAN_JAX
from trainer import Trainer
from mnist_data_loader import load_mnist_data
import numpy as np

def benchmark(model_cls, model_name, epochs):
    print(f"Running benchmark for {model_name}...")

    trainer = Trainer(model_cls=model_cls, layers_hidden=[28 * 28, 64, 10], data_loader=load_mnist_data,epochs=epochs)
    epoch_times, train_losses, val_losses, val_accuracies = trainer.train_and_evaluate()
    return {
        "model_name": model_name,
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }

if __name__ == "__main__":
    results = []

    epochs = 10

    # Benchmark EfficientKAN JAX
    results.append(benchmark(EfficientKAN_JAX, "EfficientKAN JAX",epochs=epochs))

    # Benchmark FastKAN JAX
    results.append(benchmark(FastKAN_JAX, "FastKAN JAX",epochs=epochs))

    for result in results:
        print(f"Benchmarking {result['model_name']}")
        print(f"Average Epoch Time: {np.mean(result['epoch_times']):.2f}s")
        print(f"Final Training Loss: {result['train_losses'][-1]:.4f}")
        print(f"Final Validation Loss: {result['val_losses'][-1]:.4f}")
        print(f"Final Validation Accuracy: {result['val_accuracies'][-1]:.4f}\n")
