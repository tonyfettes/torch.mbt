pub fn constant_(tensor : @tensor.Tensor, value : Float) -> Unit {
  for i in 0..<tensor.value.length() {
    tensor.value[i] = value
  }
}

pub fn ones_(tensor : @tensor.Tensor) -> Unit {
  constant_(tensor, 1.0)
}

pub fn zeros_(tensor : @tensor.Tensor) -> Unit {
  constant_(tensor, 0.0)
}
