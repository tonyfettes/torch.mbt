///|
pub fn Tensor::reshape[Shape : @shape.Shape](
  self : Tensor[Float],
  shape : Shape
) -> Tensor[Float] {
  let shape = shape.to_fixed_array()
  let block = compute_block_size(shape)
  Tensor::{
    value: self.value,
    shape,
    block,
    refcnt: 0,
    graph: Nop(self),
    grad: self.grad,
  }
}
