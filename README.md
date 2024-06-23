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
python mnist_efficient_kan_jax.py
```

This will download the MNIST dataset the first time it is run and then start training the model, displaying the training and validation progress.

### FastKAN JAX port

In additition, was also ported Ziyao Li's [FastKAN](https://github.com/ZiyaoLi/fast-kan) to JAX.

```bash
python mnist_fastkan_jax.py
```

### Benchmark

To compare the performance of the EfficientKAN and FastKAN models, we ran a benchmark on the MNIST dataset. The models were trained for 10 epochs with a batch size of 64. Below are the results: (Mac Book Pro, M2)

```bash
Benchmarking EfficientKAN
Average Epoch Time: 11.81s
Final Training Loss: 0.0122
Final Validation Loss: 0.1102
Final Validation Accuracy: 0.9706

Benchmarking FastKAN
Average Epoch Time: 7.34s
Final Training Loss: 0.0002
Final Validation Loss: 0.1180
Final Validation Accuracy: 0.9723
```

The benchmark can be run with the following command:

```bash
python benchmark.py
```

## Changelog

* 2024.06.23
  * Added FastKAN JAX port to repo.
  * Benchmark added
* 2024.06.22
  * Initial repository setup and first commit.

## License

MIT
