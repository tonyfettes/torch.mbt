pub fn Tensor::op_add(self : Tensor, other : Tensor) -> Tensor {
  let (self, other) = Tensor::broadcast(self, other)
  let length = self.value.length()
  let value : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < length; i = i + 1 {
    value[i] = self.value[i] + other.value[i]
  }
  Tensor::{
    value,
    shape: self.shape,
    block: self.block,
    graph: Graph::Add(self, other),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}
