///|
pub fn abs(self : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Float] = FixedArray::make(length, (0.0 : Float))
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = if self.value[i] < 0.0 { -self.value[i] } else { self.value[i] }
  }
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
    graph: Graph::Abs(self),
    refcnt: 0,
    grad: FixedArray::make(length, (0.0 : Float)),
  }
}
