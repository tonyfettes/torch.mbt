///|
trait TensorAdd {
  op_add(Tensor[Self], Tensor[Self]) -> Tensor[Self]
}

///|
pub impl TensorAdd for Int with op_add(self : Tensor[Int], other : Tensor[Int]) -> Tensor[
  Int,
] {
  let length = self.value.length()
  let value = FixedArray::make(length, 0)
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

///|
pub fn Tensor::op_add[Dtype : TensorAdd](
  self : Tensor[Dtype],
  other : Tensor[Dtype]
) -> Tensor[Dtype] {
  let (self, other) = Tensor::broadcast(self, other)
  TensorAdd::op_add(self, other)
}
