pub fn Tensor::fill_(self : Tensor, value : TensorValue) -> Unit {
  let other = value.to_tensor(requires_grad=false)
  @debug.assert(self.shape == other.shape, "Inconsistent shape")
  for i in 0..<self.value.length() {
    self.value[i] = other.value[i]
  }
}
