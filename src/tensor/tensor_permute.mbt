pub fn Tensor::permute(self : Tensor, permutation : Array[Int]) -> Tensor {
  if permutation.length() != self.shape.length() {
    abort("Permutation length does not match tensor rank")
  }
  let shape = FixedArray::makei(
    self.shape.length(),
    fn(i) { self.shape[permutation[i]] },
  )
  let value = Array::new(capacity=self.value.length())
  let index = Array::new(capacity=self.value.length())
  fn build(dimension : Int, offset : Int) -> Unit {
    if dimension == self.shape.length() {
      value.push(self.value[offset])
      index.push(offset)
      return
    }
    for i = 0; i < self.shape[permutation[dimension]]; i = i + 1 {
      build(dimension + 1, offset + i * self.block[permutation[dimension]])
    }
  }

  build(0, 0)
  Tensor::{
    value: FixedArray::from_array(value),
    shape,
    block: compute_block_size(shape),
    graph: Graph::Get(self, FixedArray::from_array(index)),
    ref: 0,
    grad: FixedArray::make(value.length(), 0.0),
  }
}
