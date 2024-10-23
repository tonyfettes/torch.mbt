from pathlib import Path
import argparse
import json


def _read_module_version(directory: Path) -> str:
    module_path = directory / "moon.mod.json"
    if not module_path.exists():
        raise Exception(f"Error: {module_path} does not exist")
    module = json.loads(module_path.read_text())
    if "version" not in module:
        raise Exception(f"Error: {module_path} has no version")
    return module["version"]


def _should_ignore(item: Path) -> bool:
    if item.name.startswith(".") or item.name.startswith("_"):
        return True
    return False


def _sync(directory: Path, version: str):
    for item in directory.iterdir():
        if _should_ignore(item):
            continue
        if item.is_dir():
            _sync(item, version)
        if item.is_file() and item.name == "moon.mod.json":
            module = json.loads(item.read_text())
            if "version" in module:
                module_version = module["version"]
                if module_version != version:
                    module["version"] = version
                item.write_text(json.dumps(module))
            else:
                module["version"] = version
                print(f"Error: {item} has no name or version")


def sync(directory: Path):
    module_version = _read_module_version(directory)
    _sync(directory, module_version)


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)

    sync_parser = subparsers.add_parser("sync")
    sync_parser.add_argument("directory", type=Path, default=Path("."))

    args = parser.parse_args()

    sync(args.directory)
