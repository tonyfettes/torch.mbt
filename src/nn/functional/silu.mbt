pub fn silu(input : @tensor.Tensor) -> @tensor.Tensor {
  input * input.sigmoid()
}
