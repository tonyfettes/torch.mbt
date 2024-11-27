pub(readonly) trait ToTensor: Show {
  to_tensor(Self) -> Tensor
}

pub impl ToTensor for Bool with to_tensor(self : Bool) -> Tensor {
  Tensor::{
    value: [if self { 1.0 } else { 0.0 }],
    shape: [],
    block: [],
    graph: Graph::Val,
    refcnt: 0,
    grad: [0.0],
  }
}

pub impl Default for Bool with default() -> Bool { false }

pub impl ToTensor for Int with to_tensor(self : Int) -> Tensor {
  Tensor::{
    value: [self.to_float()],
    shape: [],
    block: [],
    graph: Graph::Val,
    refcnt: 0,
    grad: [0.0],
  }
}

pub impl Default for Int with default() -> Int { 0 }

pub impl ToTensor for Float with to_tensor(self : Float) -> Tensor {
  Tensor::{
    value: [self],
    shape: [],
    block: [],
    graph: Graph::Val,
    refcnt: 0,
    grad: [0.0],
  }
}

pub impl Default for Float with default() -> Float { 0.0 }

pub impl ToTensor for Double with to_tensor(self : Double) -> Tensor {
  Tensor::{
    value: [self.to_float()],
    shape: [],
    block: [],
    graph: Graph::Val,
    refcnt: 0,
    grad: [0.0],
  }
}

pub impl Default for Double with default() -> Double { 0.0 }

pub impl[X : ToTensor + Default] ToTensor for FixedArray[X] with to_tensor(
  self : FixedArray[X]
) -> Tensor {
  let tensors = self.map(fn { x => x.to_tensor() })
  let element = if self.length() == 0 {
    X::default().to_tensor()
  } else {
    tensors[0]
  }
  let shape = FixedArray::make(element.shape.length() + 1, 0)
  shape[0] = self.length()
  for i, s in element.shape {
    shape[i + 1] = s
  }
  let block = compute_block_size(shape)
  let value : FixedArray[Float] = FixedArray::make(shape[0] * block[0], 0.0)
  for i = 0; i < tensors.length(); i = i + 1 {
    for j = 0; j < tensors[i].value.length(); j = j + 1 {
      value[i * block[0] + j] = tensors[i].value[j]
    }
  }
  Tensor::{
    value,
    shape,
    block,
    graph: Graph::Val,
    refcnt: 0,
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}

pub impl[X] Default for FixedArray[X] with default() -> FixedArray[X] { [] }

pub impl[X : ToTensor + Default] ToTensor for Array[X] with to_tensor(
  self : Array[X]
) -> Tensor {
  let tensors = self.map(fn { x => x.to_tensor() })
  let element = if self.length() == 0 {
    X::default().to_tensor()
  } else {
    tensors[0]
  }
  let shape = FixedArray::make(element.shape.length() + 1, 0)
  shape[0] = self.length()
  for i, s in element.shape {
    shape[i + 1] = s
  }
  let block = compute_block_size(shape)
  let value : FixedArray[Float] = FixedArray::make(shape[0] * block[0], 0.0)
  for i = 0; i < tensors.length(); i = i + 1 {
    for j = 0; j < tensors[i].value.length(); j = j + 1 {
      value[i * block[0] + j] = tensors[i].value[j]
    }
  }
  Tensor::{
    value,
    shape,
    block,
    graph: Graph::Val,
    refcnt: 0,
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}

pub impl[X] Default for Array[X] with default() -> Array[X] { [] }

pub impl ToTensor for Tensor with to_tensor(self : Tensor) -> Tensor { self }
