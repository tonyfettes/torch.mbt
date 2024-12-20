///|
pub impl TensorAdd for Float with op_add(
  self : Tensor[Float],
  other : Tensor[Float]
) -> Tensor[Float] {
  let length = self.value.length()
  let value = FixedArray::make(length, (0.0 : Float))
  for i = 0; i < length; i = i + 1 {
    value[i] = self.value[i] + other.value[i]
  }
  Tensor::{
    value,
    shape: self.shape,
    block: self.block,
    graph: Add(self, other),
    refcnt: 0,
    grad: FixedArray::make(length, (0.0 : Float)),
  }
}
