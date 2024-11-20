pub fn Tensor::op_set[X : TensorValue](
  self : Tensor,
  index : Int,
  other : X
) -> Unit {
  if not(no_grad.val) {
    match self.graph {
      Val => ()
      _ =>
        abort(
          "a view of a leaf Variable that requires grad is being used in an in-place operation.",
        )
    }
  }
  let other = other.to_tensor(requires_grad=false)
  let offset = index * self.block[0]
  for i = 0; i < self.block[0]; i = i + 1 {
    self.value[offset + i] = other.value[i]
  }
}
