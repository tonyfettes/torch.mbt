///|
pub fn Tensor::fill_[Value : ToTensor](self : Tensor[Float], value : Value) -> Unit {
  if @debug.DEBUG && not(no_grad.val) {
    match self.graph {
      Val => ()
      _ =>
        abort(
          "a view of a leaf Variable that requires grad is being used in an in-place operation.",
        )
    }
  }
  let other = value.to_tensor()
  if @debug.DEBUG {
    if self.shape != other.shape {
      abort("Inconsistent shape")
    }
  }
  for i in 0..<self.value.length() {
    self.value[i] = other.value[i]
  }
}
