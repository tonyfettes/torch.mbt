# tonyfettes/torch

PyTorch-like tensor operations implemented in MoonBit.

## Usage

1. Add `tonyfettes/torch` module as a dependencies

   ```bash
   moon add tonyfettes/torch
   ```

2. Import `tonyfettes/torch` package.

   ```json
   {
     "import": {
       "tonyfettes/torch"
     }
   }
   ```

3. Use it in your package

   ```moonbit
   fn main {
     let tensor = @torch.tensor(1.0)
     println("\{tensor}")
   }
   ```

## Roadmap

- [x] Use `Tensor` for everything, remove `Value` type
- [x] Use `Shape`, `Shape2d`, etc. for passing shape-like object (`kernel_size`, etc.)
- [ ] PyTorch tensor operations
  - [x] `torch.stack`
  - [x] `torch.sum` along any dimension
  - [x] `torch.cat` along any dimension
  - [ ] `torch.transpose`
  - [x] `torch.swapaxis` (implemented by `Tensor::permute` and `Tensor::moveaxis`)
  - [x] `torch.sigmoid`
  - [x] ~~Broadcast addition, multiplication, etc.~~ `torch.broadcast`
  - [x] `torch.zeros`
  - [x] `torch.ones`
- [ ] PyTorch module
  - [x] `Sigmoid`
  - [x] `AvgPool2d`
  - [ ] [`torch.nn.LogSoftmax`](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html)
- [ ] PyTorch functional module
- [ ] Module/model saving and loading
  - [x] Static JSON format.
  - [ ] Dynamic JSON format. "Dynamic" means saving/loading a trait object (`Module` here)
  - [ ] Binary format
- [ ] Computation graph optimization
  - [ ] `Get` operation fusing
  - [ ] Make `Sum` operate on `Array[Tensor]` directly
  - [ ] Eliminate `Cat(Get(...))`
- [ ] Loss
  - [ ] KL-divergence
  - [ ] [`torch.nn.NLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
- [ ] Distributions re-parametrization trick
- [ ] `Adam*` optimizers
- [ ] WASM-SIMD `v128`
  - [x] `add`
  - [x] `mul`
  - [ ] ~~`matmul`~~ [General Matrix Multiply](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3)
