pub fn Tensor::op_add(self : Tensor, other : Tensor) -> Tensor {
  let (self, other) = Tensor::broadcast(self, other)
  let length = self.value.length()
  let value = FixedArray::make(length, (0.0 : Float))
  for i = 0; i < length; i = i + 1 {
    value[i] = self.value[i] + other.value[i]
  }
  Tensor::{
    value,
    shape: self.shape,
    block: self.block,
    graph: Graph::Add(self, other),
    refcnt: 0,
    grad: FixedArray::make(length, (0.0 : Float)),
  }
}
