from efficient_kan_jax import KAN as EfficientKAN
from trainer import Trainer
from mnist_data_loader import load_mnist_data

if __name__ == "__main__":
    # Specify the model type ('EfficientKAN' or 'FastKAN')
    model_cls = EfficientKAN  # Change to FastKAN for the FastKAN model

    trainer = Trainer(model_cls=model_cls, layers_hidden=[28 * 28, 64, 10], data_loader=load_mnist_data)
    trainer.train_and_evaluate()
