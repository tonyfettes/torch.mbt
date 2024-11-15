fn Tensor::ones[Shape : @shape.Shape](
  shape : Shape,
  requires_grad~ : Bool = false
) -> Tensor {
  let shape = shape.to_fixed_array()
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value : FixedArray[Float] = FixedArray::make(total, 1.0)
  Tensor::{
    shape,
    block,
    value,
    refcnt: 0,
    graph: if requires_grad {
      Graph::Var
    } else {
      Graph::Val
    },
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
