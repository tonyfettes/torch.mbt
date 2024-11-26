pub fn Tensor::fill_(self : Tensor, value : TensorValue) -> Unit {
  let other = value.to_tensor(requires_grad=false)
  if @debug.DEBUG {
    if self.shape != other.shape {
      abort("Inconsistent shape")
    }
  }
  for i in 0..<self.value.length() {
    self.value[i] = other.value[i]
  }
}
