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

- [ ] Monofy Tensor for every dtype (`bool`, `i8`, `i16`, `i32`, `f32`, `f64`)
- [ ] PyTorch tensor operations
  - [ ] `torch.transpose`
  - [ ] `torch.argmax`
- [ ] PyTorch module
  - [ ] [`torch.nn.LogSoftmax`](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html)
  - [ ] [`torch.nn.RMSNorm`](https://pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html)
  - [ ] [`torch.nn.LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [ ] `torch.nn.init`
  - [x] `torch.nn.init.uniform_`
  - [x] `torch.nn.init.normal_`
  - [ ] `torch.nn.init.xavier_uniform_`
- [ ] Negative index support for view, etc.
- [ ] PyTorch functional module
  - [x] `torch.nn.functional.rms_norm`
  - [x] `torch.nn.functional.silu`
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
- [ ] Distributions
  - [ ] re-parametrization trick
  - [ ] `entropy`
  - [ ] `Normal::log_prob`
- [ ] `Adam*` optimizers
- [ ] WASM-SIMD `v128`
  - [x] `add`
  - [x] `mul`
  - [ ] ~~`matmul`~~ [General Matrix Multiply](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3)
