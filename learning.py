# pyright: strict

import copy

import torch
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.transforms
from tqdm import tqdm

type DataLoader = torch.utils.data.DataLoader[torch.Tensor]


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()  # type: ignore
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
            torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # type: ignore
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Learner:
    def __init__(self, dataset_str: str):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        self.dataset_str = dataset_str
        if dataset_str == "mnist":
            get_datasets = torchvision.datasets.MNIST
            self.loss_fn = torch.nn.NLLLoss()
            self.epochs = 5
        elif dataset_str == "fashionmnist":
            get_datasets = torchvision.datasets.FashionMNIST
            self.loss_fn = torch.nn.NLLLoss()
            self.epochs = 5
        elif dataset_str == "cifar10":
            get_datasets = torchvision.datasets.CIFAR10
            self.loss_fn = torch.nn.CrossEntropyLoss()
            self.epochs = 20
        else:
            raise ValueError("Unknown dataset string: {}".format(dataset_str))

        self.training_data = get_datasets(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        self.test_data = get_datasets(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

        self.train_dataloader: DataLoader = torch.utils.data.DataLoader(
            dataset=self.training_data
        )
        self.test_dataloader: DataLoader = torch.utils.data.DataLoader(
            dataset=self.test_data
        )

        self.train_list = [
            (x.to(self.device), y.to(self.device))
            for x, y in self.train_dataloader
        ]
        self.test_list = [
            (x.to(self.device), y.to(self.device))
            for x, y in self.test_dataloader
        ]

    def get_data(self):
        return copy.deepcopy(self.train_list), copy.deepcopy(self.test_list)

    def train_model(self, remove_index: list[int] | None):
        batch_size = 64
        if remove_index is None:
            train_dataloader = self.train_dataloader
        else:
            print(f"{len(remove_index)} samples removed.")
            same_list = [
                i
                for i in range(len(self.training_data))
                if i not in remove_index
            ]
            train_dataloader: DataLoader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.Subset(self.training_data, same_list),
                batch_size=batch_size,
                shuffle=False,
            )

        model = self.new_model()

        if self.dataset_str == "mnist" or self.dataset_str == "fashionmnist":
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.001, momentum=0.9
            )

        def train(
            dataloader: DataLoader,
            model: torch.nn.Module,
            loss_fn: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
        ):
            model.train()
            for _, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)

                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def test(model: torch.nn.Module):
            size = len(self.test_list)
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for x, y in self.test_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = model(x)
                    test_loss += self.loss_fn(pred, y).item()
                    correct += (
                        (pred.argmax(1) == y).type(torch.float).sum().item()
                    )
            test_loss /= size
            correct /= size
            print("Test Error:")
            print(
                f" Accuracy: {(100*correct):>0.1f}%, "
                f" Avg loss: {test_loss:>8f}"
            )

        for _ in tqdm(range(self.epochs)):
            train(train_dataloader, model, self.loss_fn, optimizer)
        test(model)
        print("Done!")
        return model

    def new_model(self) -> torch.nn.Module:
        if self.dataset_str == "mnist" or self.dataset_str == "fashionmnist":
            return NeuralNetwork().to(self.device)
        else:
            return CNN().to(self.device)


def test_accuracy(
    model: torch.nn.Module, test_data: list[tuple[torch.Tensor, int]]
):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_data:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += 1
    return correct / total
