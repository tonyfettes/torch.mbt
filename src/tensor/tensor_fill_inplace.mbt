pub fn Tensor::fill_inplace(self : Tensor, value : TensorValue) -> Unit {
  let other = value.to_tensor(requires_grad=false)
  if self.shape != other.shape {
    abort("Inconsistent shape")
  }
  for i in 0..<self.value.length() {
    self.value[i] = other.value[i]
  }
}
