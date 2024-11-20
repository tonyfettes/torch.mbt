trait TensorValue: Show {
  to_tensor(Self, requires_grad~ : Bool) -> Tensor
}

pub impl TensorValue for Float with to_tensor(
  self : Float,
  requires_grad~ : Bool
) -> Tensor {
  Tensor::{
    value: [self],
    shape: [],
    block: [],
    graph: if requires_grad {
      Graph::Var
    } else {
      Graph::Val
    },
    refcnt: 0,
    grad: [0.0],
  }
}

pub impl Default for Float with default() -> Float { 0.0 }

pub impl TensorValue for Double with to_tensor(
  self : Double,
  requires_grad~ : Bool
) -> Tensor {
  Tensor::{
    value: [self.to_float()],
    shape: [],
    block: [],
    graph: if requires_grad {
      Graph::Var
    } else {
      Graph::Val
    },
    refcnt: 0,
    grad: [0.0],
  }
}

pub impl Default for Double with default() -> Double { 0.0 }

pub impl[X : TensorValue + Default] TensorValue for FixedArray[X] with to_tensor(
  self : FixedArray[X],
  requires_grad~ : Bool
) -> Tensor {
  let tensors = self.map(fn { x => x.to_tensor(requires_grad~) })
  let element = if self.length() == 0 {
    X::default().to_tensor(requires_grad~)
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
    graph: if requires_grad {
      Graph::Var
    } else {
      Graph::Val
    },
    refcnt: 0,
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}

pub impl[X : TensorValue + Default] TensorValue for Array[X] with to_tensor(
  self : Array[X],
  requires_grad~ : Bool
) -> Tensor {
  let tensors = self.map(fn { x => x.to_tensor(requires_grad~) })
  let element = if self.length() == 0 {
    X::default().to_tensor(requires_grad~)
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
    graph: if requires_grad {
      Graph::Var
    } else {
      Graph::Val
    },
    refcnt: 0,
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}

pub impl[X] Default for Array[X] with default() -> Array[X] { [] }

pub impl[X] Default for FixedArray[X] with default() -> FixedArray[X] { [] }
