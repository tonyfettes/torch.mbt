///|
pub fn silu(input : @tensor.Tensor[Float]) -> @tensor.Tensor[Float] {
  input * input.sigmoid()
}
