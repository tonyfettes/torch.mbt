///|
pub fn Tensor::op_neg(self : Tensor[Float]) -> Tensor[Float] {
  let length = self.value.length()
  let output : FixedArray[Float] = FixedArray::make(length, (0.0 : Float))
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = -self.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
    graph: Graph::Neg(self),
    refcnt: 0,
    grad: FixedArray::make(length, (0.0 : Float)),
  }
}
