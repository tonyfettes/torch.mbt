pub fn Tensor::reshape(self : Tensor, shape : FixedArray[Int]) -> Tensor {
  let block = compute_block_size(shape)
  Tensor::{
    value: self.value,
    shape,
    block,
    ref: 0,
    graph: Nop(self),
    grad: self.grad,
  }
}
