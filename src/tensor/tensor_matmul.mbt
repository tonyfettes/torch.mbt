pub fn Tensor::matmul(self : Tensor, other : Tensor) -> Tensor {
  let self_shape = self.shape
  let other_shape = other.shape
  let self_value = self.value
  let other_value = other.value
  if self_shape.length() != 2 || other_shape.length() != 2 {
    abort("Cannot matrix multiply non-matrices")
  }
  if self_shape[1] != other_shape[0] {
    abort(
      "Matrix dimensions do not match: \{self_shape[0]}x\{self_shape[1]} * \{other_shape[0]}x\{other_shape[1]}",
    )
  }
  let output : FixedArray[Double] = FixedArray::make(
    self_shape[0] * other_shape[1],
    0.0,
  )
  for i = 0; i < self_shape[0]; i = i + 1 {
    for j = 0; j < other_shape[1]; j = j + 1 {
      for k = 0; k < self_shape[1]; k = k + 1 {
        output[i * other_shape[1] + j] += self_value[i * self_shape[1] + k] *
          other_value[k * other_shape[1] + j]
      }
    }
  }
  Tensor::{
    value: output,
    shape: [self_shape[0], other_shape[1]],
    block: [other_shape[1], 1],
    graph: Graph::MatMul(self, other),
    ref: 0,
    grad: FixedArray::make(self_shape[0] * other_shape[1], 0.0),
  }
}
