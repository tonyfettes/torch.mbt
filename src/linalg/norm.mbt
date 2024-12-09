pub fn vector_norm(input : @tensor.Tensor) -> @tensor.Tensor {
  input.pow(2).sum().sqrt()
}
