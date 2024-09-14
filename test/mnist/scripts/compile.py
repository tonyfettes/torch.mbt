from pathlib import Path
import pydantic
import argparse
import json
from abc import abstractmethod


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
            (dest / f"{name}_{i:05d}.json").write_text(data.model_dump_json())

    def compile(self, dest: Path, train: list[Data], test: list[Data]):
        self.compile_dataset(dest, "train", train)
        self.compile_dataset(dest, "test", test)


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
    train_dataset: list[Data] = []
    for line in json.loads((input / "train.json").read_text()):
        data = Data.model_validate(line)
        train_dataset.append(data)
    test_dataset: list[Data] = []
    for line in json.loads((input / "test.json").read_text()):
        data = Data.model_validate(line)
        test_dataset.append(data)

    compiler.compile(dest=args.output, train=train_dataset, test=test_dataset)


if __name__ == "__main__":
    main()
