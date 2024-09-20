# tonyfettes/torch

PyTorch-like API implemented in MoonBit.

## Usage

1. Add `tonyfettes/torch` module as a dependencies

   ```bash
   moon add tonyfettes/torch
   ```

2. Import `tonyfettes/torch` package.

   ```json
   {
     // ...
     "import": {
       "tonyfettes/torch"
     }
   }
   ```

3. Use it in your package

   ```moonbit
   fn main {
     tensor = @torch.tensor(1.0)
     println("\{tensor}")
   }
   ```

## Roadmap

- [x] Use `Tensor` for everything, remove `Value` type
- [ ] Use `ToIntArray` for passing shape-like object (`kernel_size`, etc.)
- [ ] PyTorch tensor operations
  - [ ] `torch.stack`
  - [ ] `torch.sum` along any dimension
  - [ ] `torch.cat` along any dimension
  - [ ] `torch.transpose`
  - [ ] `torch.swapaxis`
  - [ ] `torch.sigmoid`
  - [ ] Broadcast addition, multiplication, etc.
- [ ] PyTorch module
  - [ ] `Sigmoid`
- [ ] PyTorch functional module
  - [ ] `F.relu`
  - [ ] `F.conv2d`
  - [ ] `F.max_pool2d`
  - [ ] `F.avg_pool2d`
- [ ] Module/model saving and loading
- [ ] Computation graph optimization
  - [ ] `Get` operation fusing
  - [ ] Make `Sum` operate on `Array[Tensor]` directly
  - [ ] Eliminate `Cat(Get(...))`
- [ ] Loss
  - [ ] KL-divergence
- [ ] Distributions reparametrisation trick
- [ ] `Adam*` optimizers
- [ ] WASM-SIMD `v128`
