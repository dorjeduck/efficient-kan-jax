## efficient-kan-jax

This project is a port of Blealtan's [efficient-kan](https://github.com/Blealtan/efficient-kan) to [JAX](https://github.com/google/jax).

## How to Use

We have ported the `mnist.py` example to use our JAX-based KAN implementation.

### Install Requirements

```bash
pip install -r requirements.txt
``` 

This will install JAX, Optax, Flax, PyTorch, Torchvision, and TQDM.


### Running the MNIST Example

After installing the dependencies, you can run the MNIST example using the following command:

```bash
python mnist.py
``` 

This will download the MNIST dataset the first time it is run and then start training the model, displaying the training and validation progress.

## Changelog

* 2024.06.22
    * IInitial repository setup and first commit.

## License

MIT