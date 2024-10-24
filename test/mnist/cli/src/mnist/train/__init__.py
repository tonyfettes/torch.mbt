import mnist.model
import mnist.refer
import wasmtime
from pathlib import Path
import gzip
import json
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typing import TypedDict

writer = SummaryWriter()


class Data(TypedDict):
    image: list[int]
    label: int


class PyTorchTrainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.CrossEntropyLoss

    def __init__(self, model: torch.nn.Module, learning_rate: float):
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, batch: list[Data]) -> float:
        batch_size = len(batch)
        input = (
            torch.tensor([data["image"] for data in batch])
            .float()
            .reshape(batch_size, 1, 28, 28)
        )
        label = torch.tensor([data["label"] for data in batch])
        self.optimizer.zero_grad()
        output = self.model(input)
        loss: torch.Tensor = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class MoonBitTrainer:
    store: wasmtime.Store
    model: mnist.model.Root
    learning_rate: float

    def __init__(self, learning_rate: float):
        self.store = wasmtime.Store()
        self.model = mnist.model.Root(self.store)
        self.learning_rate = learning_rate

    def train(self, batch: list[Data]) -> float:
        input = [(bytes(data["image"]), data["label"]) for data in batch]
        return self.model.train(self.store, input, self.learning_rate)

def main():
    test_path = Path("data/mnist_handwritten_test.json.gz")
    train_path = Path("data/mnist_handwritten_train.json.gz")

    with gzip.open(test_path, "rt") as test_file:
        test_dataset = json.load(test_file)

    with gzip.open(train_path, "rt") as train_file:
        train_dataset = json.load(train_file)

    learning_rate = 0.001
    batch_size = 8

    moonbit_trainer = MoonBitTrainer(learning_rate)
    pytorch_trainer = PyTorchTrainer(mnist.refer.model, learning_rate)

    for i in range(0, len(train_dataset), batch_size):
        batch = train_dataset[i : i + batch_size]

        moonbit_loss = moonbit_trainer.train(batch)
        pytorch_loss = pytorch_trainer.train(batch)
        print(f"> torch.mbt: {moonbit_loss}")
        writer.add_scalar("MoonBit/loss", moonbit_loss, i)

        print(f"> PyTorch  : {pytorch_loss}")
        writer.add_scalar("PyTorch/loss", pytorch_loss, i)

        writer.add_scalar("diff", abs(moonbit_loss - pytorch_loss), i)

    writer.flush()
    writer.close()
