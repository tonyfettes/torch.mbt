import mnist.model
import mnist.refer
from wasmtime import Store
import requests
from pathlib import Path
import gzip
import json


def main():
    store = Store()
    model = mnist.model.Root(store)

    test_path = Path("data/mnist_handwritten_test.json.gz")
    train_path = Path("data/mnist_handwritten_train.json.gz")

    with gzip.open(test_path, "rt") as test_file:
        test_dataset = json.load(test_file)

    with gzip.open(train_path, "rt") as train_file:
        train_dataset = json.load(train_file)

    for train_data in train_dataset:
        input = bytes(train_data["image"])
        print(model.infer(store, input))
        break
