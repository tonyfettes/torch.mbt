# MNIST Web & CLI Application

This demo application includes a web page and a command-line tool to train and infer a model on the MNIST dataset.

The application is built around a computation core
written in MoonBit, which communicates with external JavaScript/Python through the [WebAssembly Component Model](https://component-model.bytecodealliance.org/)

## Prepare

You need to have the following tools from the WebAssembly toolchain installed:

- [`wit-bindgen`](https://github.com/bytecodealliance/wit-bindgen)
- [`wasm-tools`](https://github.com/bytecodealliance/wasm-tools)

We are using [`jco`](https://github.com/bytecodealliance/jco) and [`wasmtime-py`](https://github.com/bytecodealliance/wasmtime-py) to help generate host glue code. You don't need install these manually, as them will be handled by package manager of each language.

## Web

```bash
pnpm install
python3 scripts/build.py --target=wasm --profile=release web
pnpm run build web/
pnpm run preview web/
```

Then open the URL prompt in your terminal (it should looks like <http://localhost:4173>)

## CLI

A Python (3.11 or greater) virtual environment is recommended.

```bash
pip3 install -e '.[dev]'
```

The command above install an editable installation to the working directory, which means:

1. You can use `mnist` command to call the `cli.mnist.main`.
2. You can tweak with the Python code to see the change.

However, before executing the program, you need to download the MNIST dataset.

```bash
curl -fsSL "https://media.githubusercontent.com/media/lorenmh/mnist_handwritten_json/master/mnist_handwritten_test.json.gz" -o data/mnist_handwritten_test.json.gz
curl -fsSL "https://media.githubusercontent.com/media/lorenmh/mnist_handwritten_json/master/mnist_handwritten_train.json.gz" -o data/mnist_handwritten_train.json.gz
```

Then, you can start train/evaluate a model over the MNIST dataset. See the help message of the command for more usage.

```bash
# Warning: this will not load/save any models.
mnist train
```

### Deno Implementation

We provide a Deno implementation of the cli program.

- Fetch data.

  ```bash
  deno --unstable-sloppy-imports --unstable-webgpu --allow-read --allow-net --allow-write cli/index.ts fetch data
  ```

- Train.

  ```bash
  deno --unstable-sloppy-imports --unstable-webgpu --allow-read --allow-net --allow-write cli/index.ts train --data data
  ```
