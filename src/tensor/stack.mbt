///|
pub fn Tensor::stack[Dtype : @dtype.Dtype](
  tensors : FixedArray[Tensor[Dtype]]
) -> Tensor[Dtype] {
  if @debug.DEBUG {
    if tensors.length() == 0 {
      abort("Cannot stack an empty list of tensors")
    }
  }
  let element = tensors[0]
  let shape = @unsafe.UnsafeArray::make(element.shape.length() + 1, 0)
  shape[0] = tensors.length()
  for i in 0..<element.shape.length() {
    shape[i + 1] = element.shape[i]
  }
  let block = compute_block_size(shape._)
  let total = shape[0] * block[0]
  let value = @unsafe.UnsafeArray::new(total)
  let index = FixedArray::make(total, (-1, -1))
  for i in 0..<tensors.length() {
    if @debug.DEBUG {
      if tensors[i].shape != element.shape {
        abort("Cannot stack tensors with different shapes")
      }
    }
    let tensor_i_value = tensors[i].value
    for j in 0..<tensor_i_value.length() {
      let k = i * block[0] + j
      value[k] = tensor_i_value[j]
      index[k] = (i, j)
    }
  }
  Tensor::{
    value,
    shape,
    graph: Cat(tensors, index),
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
