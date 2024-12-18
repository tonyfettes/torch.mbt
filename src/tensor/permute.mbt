///|
pub fn Tensor::permute[Shape : @shape.Shape](
  self : Tensor[Float],
  permutation : Shape
) -> Tensor[Float] {
  let self_shape = self.shape
  let self_block = self.block()
  let permutation = permutation.to_fixed_array()
  if permutation.length() != self_shape.length() {
    abort("Permutation length does not match tensor rank")
  }
  let shape = FixedArray::makei(self_shape.length(), fn(i) {
    self_shape[permutation[i]]
  })
  let value = Array::new(capacity=self.value.length())
  let index = Array::new(capacity=self.value.length())
  fn build(dimension : Int, offset : Int) -> Unit {
    if dimension == self_shape.length() {
      value.push(self.value[offset])
      index.push(offset)
      return
    }
    for i = 0; i < self_shape[permutation[dimension]]; i = i + 1 {
      build(dimension + 1, offset + i * self_block[permutation[dimension]])
    }
  }

  build(0, 0)
  Tensor::{
    value: FixedArray::from_array(value),
    shape,
    graph: Graph::Get(self, FixedArray::from_array(index)),
    refcnt: 0,
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}
