///|
pub fn Tensor::reshape[Shape : @shape.Shape](
  self : Tensor[Float],
  shape : Shape
) -> Tensor[Float] {
  let shape = shape.to_fixed_array()
  Tensor::{
    value: self.value,
    shape,
    refcnt: 0,
    graph: Nop(self),
    grad: self.grad,
  }
}
