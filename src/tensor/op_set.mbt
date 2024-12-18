///|
pub fn Tensor::op_set[Value : ToValue](
  self : Tensor[Float],
  index : Int,
  other : Value
) -> Unit {
  let self_block = self.block()
  if @debug.DEBUG && not(no_grad.val) {
    match self.graph {
      Val => ()
      _ =>
        abort(
          "a view of a leaf Variable that requires grad is being used in an in-place operation.",
        )
    }
  }
  let other = ValueToTensor::value_to_tensor(other.to_value())
  let offset = index * self_block[0]
  for i = 0; i < self_block[0]; i = i + 1 {
    self.value[offset + i] = other.value[i]
  }
}
