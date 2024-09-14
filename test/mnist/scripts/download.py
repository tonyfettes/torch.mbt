from pathlib import Path
import pydantic
import argparse
import json
from abc import abstractmethod
from tqdm import tqdm
import requests
import gzip


class Data(pydantic.BaseModel):
    image: list[int]
    label: int


class Compiler:
    @abstractmethod
    def compile(self, dest: Path, train: list[Data], test: list[Data]): ...


class MoonbitCompiler(Compiler):
    def compile_types(self, dest: Path):
        (dest / "types.mbt").write_text(
            f"""pub struct MnistData {{
  image: Array[Int]
  label: Int
}}"""
        )

    def compile_data(self, dest: Path, name: str, data: Data):
        dest.write_text(
            f"let {name} : MnistData = {{"
            f"  image: {data.image},"
            f"  label: {data.label}"
            f"}}"
        )

    def compile_dataset(self, dest: Path, name: str, dataset: list[Data]):
        for i, data in enumerate(dataset):
            self.compile_data(dest / f"{name}_{i:05d}.mbt", f"{name}_{i:05d}", data)
        lines = [f"pub let {name} : Array[MnistData] = ["]
        for i in range(len(dataset)):
            lines.append(f"  {name}_{i:05d},")
        lines.append("]")
        (dest / f"{name}.mbt").write_text("\n".join(lines))

    def compile(self, dest: Path, train: list[Data], test: list[Data]):
        self.compile_types(dest)
        self.compile_dataset(dest, "mnist_train", train)
        self.compile_dataset(dest, "mnist_test", test)


class JSONCompiler(Compiler):
    def compile_dataset(self, dest: Path, name: str, dataset: list[Data]):
        for i, data in enumerate(dataset):
            (dest / f"{name}_{i}.json").write_text(data.model_dump_json())

    def compile(self, dest: Path, train: list[Data], test: list[Data]):
        self.compile_dataset(dest, "train", train)
        self.compile_dataset(dest, "test", test)


def load_dataset(url: str, dest: Path) -> list[Data]:
    if not dest.exists():
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()

        total = int(response.headers.get("Content-Length", 0))

        dest.parent.mkdir(parents=True, exist_ok=True)

        with tqdm(total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
            with dest.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

    lines = gzip.decompress(dest.read_bytes()).decode().splitlines()
    dataset: list[Data] = []
    for line in lines:
        data = Data.model_validate(json.loads(line))
        dataset.append(data)

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json", "moonbit"], default="json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("input", type=Path)

    args = parser.parse_args()

    if args.format == "json":
        compiler = JSONCompiler()
    elif args.format == "moonbit":
        compiler = MoonbitCompiler()
    else:
        raise ValueError(f"Unknown format: {args.format}")

    input: Path = args.input

    train_url = "https://github.com/Eventual-Inc/mnist-json/raw/master/mnist_handwritten_train.json.gz"
    train_dataset = load_dataset(train_url, input / "train.json.gz")

    test_url = "https://github.com/Eventual-Inc/mnist-json/raw/master/mnist_handwritten_test.json.gz"
    test_dataset = load_dataset(test_url, input / "test.json.gz")

    compiler.compile(dest=args.output, train=train_dataset, test=test_dataset)


if __name__ == "__main__":
    main()
