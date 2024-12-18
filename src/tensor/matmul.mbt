///|
pub fn Tensor::matmul(
  self : Tensor[Float],
  other : Tensor[Float]
) -> Tensor[Float] {
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
  let row = self_shape[0]
  let col = other_shape[1]
  let key = self_shape[1]
  let total = row * col
  let value = @unsafe.UnsafeArray::make(total, (0.0 : Float))
  for r = 0; r < row; r = r + 1 {
    for c = 0; c < col; c = c + 1 {
      for k = 0; k < key; k = k + 1 {
        // value[r][c] += self_value[r][k] * other_value[k][c]
        value[r * col + c] += self_value[r * key + k] * other_value[k * col + c]
      }
    }
  }
  Tensor::{
    value,
    shape: [row, col],
    graph: Graph::MatMul(self, other),
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
