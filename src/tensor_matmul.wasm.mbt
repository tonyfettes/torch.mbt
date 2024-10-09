pub fn Tensor::matmul(self : Tensor, other : Tensor) -> Tensor {
  if self.shape.length() != 2 || other.shape.length() != 2 {
    abort("Cannot matrix multiply non-matrices")
  }
  if self.shape[1] != other.shape[0] {
    abort(
      "Matrix dimensions do not match: \{self.shape[0]}x\{self.shape[1]} * \{other.shape[0]}x\{other.shape[1]}",
    )
  }
  let output : FixedArray[Double] = FixedArray::make(
    self.shape[0] * other.shape[1],
    0.0,
  )
  for i = 0; i < self.shape[0]; i = i + 1 {
    for j = 0; j < other.shape[1]; j = j + 1 {
      for k = 0; k < self.shape[1]; k = k + 1 {
        output[i * other.shape[1] + j] += self.value[i * self.shape[1] + k] *
          other.value[k * other.shape[1] + j]
      }
    }
  }
  Tensor::{
    value: output,
    shape: [self.shape[0], other.shape[1]],
    block: [other.shape[1], 1],
    graph: Graph::MatMul(self, other),
    ref: 0,
    grad: FixedArray::make(self.shape[0] * other.shape[1], 0.0),
  }
}
