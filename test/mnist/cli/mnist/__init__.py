import wasmtime
import mnist.model
from typing import TypedDict
from pathlib import Path
import gzip
import json
from torch.utils.tensorboard.writer import SummaryWriter
import argparse

writer = SummaryWriter()


class Data(TypedDict):
    image: list[int]
    label: int


class Model:
    store: wasmtime.Store
    model: mnist.model.Root

    def __init__(self):
        self.store = wasmtime.Store()
        self.model = mnist.model.Root(self.store)

    def train(self, batch: list[Data], learning_rate: float) -> float:
        input = [
            ([value / 255.0 for value in data["image"]], data["label"])
            for data in batch
        ]
        return self.model.train(self.store, input, learning_rate)

    def infer(self, image: list[int]) -> list[float]:
        return self.model.infer(self.store, [value / 255.0 for value in image])

    def save(self, path: Path):
        path.write_text(self.model.save(self.store))

    def load(self, path: Path):
        self.model.load(self.store, path.read_text())


def train(model: Model, epoch: int, batch_size: int, learning_rate: float):
    train_path = Path("data/mnist_handwritten_train.json.gz")

    with gzip.open(train_path, "rt") as train_file:
        train_dataset = json.load(train_file)

    for _ in range(epoch):
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i : i + batch_size]
            step = epoch * len(train_dataset) + i

            loss = model.train(batch, learning_rate)
            print(f"loss: {loss}")
            writer.add_scalar("loss", loss, step)  # type: ignore

    writer.flush()
    writer.close()


def eval(model: Model) -> float:
    test_path = Path("data/mnist_handwritten_test.json.gz")

    with gzip.open(test_path, "rb") as file:
        test_dataset = json.load(file)

    correct = 0
    for data in test_dataset:
        logits = model.infer(data["image"])
        prediction = logits.index(max(logits))
        if prediction == data["label"]:
            correct += 1

    accuracy = correct / len(test_dataset)
    print("accuracy: ", accuracy)
    return accuracy


def save(model: Model, path: Path):
    model.save(path)


def load(model: Model, path: Path):
    model.load(path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="command to run")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epoch", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_parser.add_argument("-o", "--output", type=Path)
    train_parser.add_argument("model", type=Path, nargs="?", default=None)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("model", type=Path)

    args = parser.parse_args()

    model = Model()
    if args.command == "train":
        if args.model:
            load(model, args.model)
        train(model, args.epoch, args.batch_size, args.learning_rate)
        save(model, args.output)
    elif args.command == "eval":
        load(model, args.model)
        eval(model)


if __name__ == "__main__":
    main()
