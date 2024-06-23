from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_mnist_data(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root="./data", train=True,
                              download=True, transform=transform)
    valset = datasets.MNIST(root="./data", train=False,
                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader
