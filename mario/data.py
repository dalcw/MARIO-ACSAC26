from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(data_dir="./data", batch_size=512, num_workers=2, download=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform,
        download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
