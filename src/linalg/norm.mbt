///|
pub fn vector_norm(input : @tensor.Tensor[Float]) -> @tensor.Tensor[Float] {
  input.pow(2).sum().sqrt()
}
