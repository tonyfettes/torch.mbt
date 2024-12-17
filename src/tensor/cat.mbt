///|
pub fn Tensor::cat(
  tensors : FixedArray[Tensor[Float]],
  dim~ : Int = 0
) -> Tensor[Float] {
  if tensors.length() == 0 {
    abort("No tensors to concatenate")
  }
  if tensors.length() == 1 {
    abort("Meaningless Tensor::cat called")
  }
  for i = 0; i < tensors.length(); i = i + 1 {
    if tensors[i].shape.length() != tensors[0].shape.length() {
      abort("Inconsistent shape")
    }
  }
  for i = 0; i < tensors[0].shape.length(); i = i + 1 {
    if i != dim && tensors[i].shape[i] != tensors[0].shape[i] {
      abort("Inconsistent shape")
    }
  }
  let shape = FixedArray::make(tensors[0].shape.length(), 0)
  for i = 0; i < tensors[0].shape.length(); i = i + 1 {
    if i == dim {
      for j = 0; j < tensors.length(); j = j + 1 {
        shape[i] += tensors[j].shape[i]
      }
    } else {
      shape[i] = tensors[0].shape[i]
    }
  }
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value = Array::new(capacity=total)
  let index = Array::new(capacity=total)
  fn build(i : Int, d : Int, p : Int) -> Unit {
    if d == tensors[i].shape.length() {
      value.push(tensors[i].value[p])
      index.push((i, p))
      return
    }
    if d == dim {
      for i = 0; i < tensors.length(); i = i + 1 {
        for j = 0; j < tensors[i].shape[d]; j = j + 1 {
          build(i, d + 1, p + j * tensors[i].block[d])
        }
      }
      return
    }
    for j = 0; j < tensors[i].shape[d]; j = j + 1 {
      build(i, d + 1, p + j * tensors[i].block[d])
    }
  }

  build(0, 0, 0)
  Tensor::{
    value: FixedArray::from_array(value),
    shape,
    block,
    graph: Cat(tensors, FixedArray::from_array(index)),
    refcnt: 0,
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}
