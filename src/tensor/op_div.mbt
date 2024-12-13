///|
pub fn Tensor::op_div(self : Tensor[Float], other : Tensor[Float]) -> Tensor[Float] {
  let (self, other) = Tensor::broadcast(self, other)
  let length = self.value.length()
  let output : FixedArray[Float] = FixedArray::make(length, (0.0 : Float))
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i] / other.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
    graph: Graph::Div(self, other),
    refcnt: 0,
    grad: FixedArray::make(length, (0.0 : Float)),
  }
}
