pub trait TensorLike {
  to_tensor(Self) -> Tensor
}

pub impl[X : TensorLike] TensorLike for Array[X] with to_tensor(self : Array[X]) -> Tensor {
  let tensors = FixedArray::from_array(self)
  TensorLike::to_tensor(tensors)
}

pub impl[X : TensorLike] TensorLike for FixedArray[X] with to_tensor(
  self : FixedArray[X]
) -> Tensor {
  let tensors = self.map(fn(x) { x.to_tensor() })
  if tensors.length() == 0 {
    abort("No tensors to concatenate")
  }
  if tensors.length() == 1 {
    return tensors[0].reshape([1, ..tensors[0].shape._])
  }
  for i in 1..<tensors.length() {
    if tensors[i].shape._ != tensors[0].shape._ {
      abort(
        "Tensor at \{i} has shape \{tensors[i].shape}, which is not the same as \{tensors[0].shape}",
      )
    }
  }
  let shape = FixedArray::make(tensors[0].shape.length() + 1, 0)
  shape[0] = tensors.length()
  for i, s in tensors[0].shape._ {
    shape[i + 1] = s
  }
  let block = compute_block_size(shape)
  let value : FixedArray[Float] = FixedArray::make(shape[0] * block[0], 0.0)
  let index = FixedArray::make(shape[0] * block[0], (-1, -1))
  for i = 0; i < tensors.length(); i = i + 1 {
    for j = 0; j < tensors[i].value.length(); j = j + 1 {
      value[i * block[0] + j] = tensors[i].value[j]
      index[i * block[0] + j] = (i, j)
    }
  }
  Tensor::{
    value,
    shape,
    block,
    graph: Cat(tensors, index),
    ref: 0,
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}

pub impl TensorLike for Tensor with to_tensor(self : Tensor) -> Tensor { self }
