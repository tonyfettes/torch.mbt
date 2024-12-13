///|
pub fn eye_(tensor : @tensor.Tensor[Float]) -> Unit {
  if @debug.DEBUG && tensor.shape.length() != 2 {
    abort("Only tensors with 2 dimensions are supported")
  }
  let h = tensor.shape[0]
  let w = tensor.shape[1]
  for i in 0..<h {
    for j in 0..<w {
      if i == j {
        tensor.value[i * w + j] = 1
      } else {
        tensor.value[i * w + j] = 0
      }
    }
  }
}
