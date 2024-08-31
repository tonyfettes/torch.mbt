pub trait Tensor: Show {
  tensor(Self) -> @tensor.Tensor
}

trait Default {
  default() -> Self
}

impl Tensor for Double with tensor(self : Double) -> @tensor.Tensor {
  @tensor.Tensor::{
    value: [self],
    shape: [],
    size: [],
    graph: @tensor.Graph::Val,
    ref: 0,
    grad: [0.0],
  }
}

impl Default for Double with default() -> Double { 0.0 }

pub fn Tensor::new[X : Tensor](self : X) -> Tensor {
  self.tensor()
}

pub fn tensor[X : Tensor](x : X) -> @tensor.Tensor {
  x.tensor()
}

impl[X : Tensor + Default] Tensor for Array[X] with tensor(self : Array[X]) -> @tensor.Tensor {
  let tensors = self.map(fn { x => x.tensor() })
  let shape = [self.length()]
  if self.length() == 0 {
    for s in X::default().tensor().shape {
      shape.push(s)
    }
  } else {
    for i = 0; i < tensors[0].shape.length(); i = i + 1 {
      shape.push(tensors[0].shape[i])
    }
  }
  let size = [1]
  for i = shape.length() - 1; i > 0; i = i - 1 {
    size.push(shape[i] * size[size.length() - 1])
  }
  size.rev_inplace()
  let value = []
  for i = 0; i < tensors.length(); i = i + 1 {
    for j = 0; j < tensors[i].value.length(); j = j + 1 {
      value.push(tensors[i].value[j])
    }
  }
  @tensor.Tensor::{
    value: FixedArray::from_array(value),
    shape: FixedArray::from_array(shape),
    size: FixedArray::from_array(size),
    graph: @tensor.Graph::Val,
    ref: 0,
    grad: FixedArray::make(value.length(), 0.0),
  }
}

impl[X] Default for Array[X] with default() -> Array[X] { [] }
