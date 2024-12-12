///|
pub fn empty[Shape : @shape.Shape](
  shape : Shape,
  requires_grad~ : Bool = false
) -> Tensor {
  let shape = shape.to_fixed_array()
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value : @unsafe.UnsafeArray[Float] = @unsafe.UnsafeArray::new(total)
  Tensor::{
    value,
    shape,
    block,
    graph: if requires_grad {
      Graph::Var
    } else {
      Val
    },
    refcnt: 0,
    grad: @unsafe.UnsafeArray::make(total, (0.0 : Float)),
  }
}
