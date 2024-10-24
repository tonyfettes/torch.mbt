pub fn Tensor::exp(self : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Float] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i].exp()
  }
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
    graph: Graph::Exp(self),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}
