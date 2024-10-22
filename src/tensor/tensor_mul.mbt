pub fn Tensor::op_mul(self : Tensor, other : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Float] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i] * other.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
    graph: Graph::Mul(self, other),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}
