pub struct Tensor {
  mut value : FixedArray[Double]
  shape : FixedArray[Int]
  size : FixedArray[Int]
  priv mut graph : Graph
  priv mut ref : Int
  mut grad : FixedArray[Double]
}

pub enum Graph {
  Val
  Var
  Add(Tensor, Tensor)
  Sub(Tensor, Tensor)
  Mul(Tensor, Tensor)
  Div(Tensor, Tensor)
  Neg(Tensor)
  Exp(Tensor)
  Log(Tensor)
  Sum(Tensor)
  Pow(Tensor, Int)
  MatMul(Tensor, Tensor)
} derive(Show)

pub trait TensorValue {
  tensor(Self) -> Tensor
}

trait Default {
  default() -> Self
}

impl TensorValue for Double with tensor(self : Double) -> Tensor {
  Tensor::{
    value: [self],
    shape: [],
    size: [],
    graph: Graph::Val,
    ref: 0,
    grad: [0.0],
  }
}

impl Default for Double with default() -> Double { 0.0 }

impl[X : TensorValue + Default] TensorValue for Array[X] with tensor(
  self : Array[X]
) -> Tensor {
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
  Tensor::{
    value: FixedArray::from_array(value),
    shape: FixedArray::from_array(shape),
    size: FixedArray::from_array(size),
    graph: Graph::Val,
    ref: 0,
    grad: FixedArray::make(value.length(), 0.0),
  }
}

impl[X] Default for Array[X] with default() -> Array[X] { [] }

pub fn Tensor::new[X : TensorValue](
  value : X,
  ~requires_grad : Bool = false
) -> Tensor {
  let tensor = value.tensor()
  if requires_grad {
    tensor.graph = Graph::Var
  }
  tensor
}

pub fn Tensor::get(self : Tensor, index : Array[Int]) -> Double {
  if index.length() != self.shape.length() {
    abort("Index length does not match tensor shape")
  }
  let mut offset = 0
  for i = 0; i < index.length(); i = i + 1 {
    if index[i] < 0 || index[i] >= self.shape[i] {
      abort("Index out of bounds")
    }
    offset += index[i] * self.size[i]
  }
  println("offset: \{offset}")
  self.value[offset]
}

fn Tensor::_output(
  self : Tensor,
  logger : Logger,
  range : Array[TensorIndex],
  index : Array[Int]
) -> Unit {
  println("index: \{index}")
  println("shape: \{self.shape}")
  if index.length() == self.shape.length() {
    logger.write_string(self.get(index).to_string())
    return
  }
  logger.write_string("[")
  let dimension = index.length()
  println("dimension: \{dimension}")
  index.push(0)
  let (start, stop) = range[dimension]
  for i = start; i < stop; i = i + 1 {
    index[dimension] = i
    self._output(logger, range, index)
    if i < self.shape[dimension] - 1 {
      logger.write_string(", ")
    }
  }
  index.pop_exn() |> ignore
  logger.write_string("]")
}

pub fn Tensor::output(self : Tensor, logger : Logger) -> Unit {
  let range = Array::makei(self.shape.length(), fn { i => (0, self.shape[i]) })
  self._output(logger, range, [])
}

pub fn Tensor::to_string(self : Tensor) -> String {
  Show::to_string(self.value)
}

test "Tensor::new - Scalar" {
  let value = 1.0
  let tensor = Tensor::new(value)
  inspect!(tensor.value, content="[1]")
  inspect!(tensor.shape, content="[]")
  inspect!(tensor.size, content="[]")
  inspect!(tensor.graph, content="Val")
  inspect!(tensor.ref, content="0")
  inspect!(tensor.grad, content="[0]")
  inspect!(tensor, content="1")
}

test "Tensor::new - Vector" {
  let value = [0.0, 1.0, 2.0, 3.0]
  let tensor = Tensor::new(value)
  inspect!(tensor.value, content="[0, 1, 2, 3]")
  inspect!(tensor.shape, content="[4]")
  inspect!(tensor.size, content="[1]")
  inspect!(tensor.graph, content="Val")
  inspect!(tensor.ref, content="0")
  inspect!(tensor.grad, content="[0, 0, 0, 0]")
  inspect!(tensor, content="[0, 1, 2, 3]")
}

test "Tensor::new - Matrix" {
  let value = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
  let tensor = Tensor::new(value)
  inspect!(tensor.value, content="[0, 1, 2, 3, 4, 5]")
  inspect!(tensor.shape, content="[2, 3]")
  inspect!(tensor.size, content="[3, 1]")
  inspect!(tensor.graph, content="Val")
  inspect!(tensor.ref, content="0")
  inspect!(tensor.grad, content="[0, 0, 0, 0, 0, 0]")
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}

test "Tensor::new - Tensor" {
  let tensor = Tensor::new(
    [
      [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
      [
        [12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0],
        [20.0, 21.0, 22.0, 23.0],
      ],
    ],
  )
  inspect!(
    tensor.value,
    content="[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]",
  )
  inspect!(tensor.shape, content="[2, 3, 4]")
  inspect!(tensor.size, content="[12, 4, 1]")
  inspect!(tensor.graph, content="Val")
  inspect!(tensor.ref, content="0")
  inspect!(
    tensor.grad,
    content="[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
  )
  inspect!(
    tensor,
    content="[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]",
  )
}

pub fn Tensor::op_get(self : Tensor, index : Int) -> TensorView {
  if self.shape.length() == 0 {
    abort("Tensor is a scalar")
  }
  TensorView::{ tensor: self, index: [(index, index + 1)] }
}

test "Tensor::op_get - Vector" {
  let tensor = Tensor::new([0.0, 1.0, 2.0, 3.0])
  let view = tensor[1]
  inspect!(view, content="[1, ]")
}

test "Tensor::op_get - Matrix" {
  let tensor = Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let view = tensor[1]
  inspect!(view, content="[3, 4, 5]")
}

pub fn Tensor::op_set(self : Tensor, index : Int, value : Tensor) -> Unit {
  for i = self.size[0] * index; i < self.size[0] * (index + 1); i = i + 1 {
    self.value[i] = value.value[i - self.size[0] * index]
  }
}

pub fn Tensor::length(self : Tensor) -> Int {
  self.shape[0]
}

pub fn TensorView::length(self : TensorView) -> Int {
  self.tensor.shape[self.index.length()]
}

pub fn Tensor::op_add(self : Tensor, other : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < length; i = i + 1 {
    output[i] = self.value[i] + other.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    size: self.size,
    graph: Graph::Add(self, other),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::op_sub(self : Tensor, other : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i] - other.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    size: self.size,
    graph: Graph::Sub(self, other),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::op_neg(self : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = -self.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    size: self.size,
    graph: Graph::Neg(self),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::op_mul(self : Tensor, other : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i] * other.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    size: self.size,
    graph: Graph::Mul(self, other),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::op_div(self : Tensor, other : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i] / other.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    size: self.size,
    graph: Graph::Div(self, other),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::exp(self : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i].exp()
  }
  Tensor::{
    value: output,
    shape: self.shape,
    size: self.size,
    graph: Graph::Exp(self),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::log(self : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i].ln()
  }
  Tensor::{
    value: output,
    shape: self.shape,
    size: self.size,
    graph: Graph::Log(self),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::sum(self : Tensor) -> Tensor {
  let length = self.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.length(); i = i + 1 {
    for j = i * self.size[0]; j < (i + 1) * self.size[0]; j = j + 1 {
      output[i] = output[i] + self.value[j]
    }
  }
  let shape : FixedArray[Int] = FixedArray::make(self.shape.length() - 1, 0)
  for i = 1; i < self.shape.length(); i = i + 1 {
    shape[i - 1] = self.shape[i]
  }
  let size : FixedArray[Int] = FixedArray::make(self.shape.length() - 1, 0)
  for i = 1; i < self.size.length(); i = i + 1 {
    size[i - 1] = self.size[i]
  }
  Tensor::{
    value: output,
    shape,
    size,
    graph: Graph::Sum(self),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::dot(self : Tensor, other : Tensor) -> Tensor {
  Tensor::sum(self * other)
}

pub enum TensorIndex {
  Index(Int)
  Range(Int, Int)
} derive(Show)

pub struct TensorView {
  tensor : Tensor
  index : Array[TensorIndex]
}

pub fn TensorView::op_get(self : TensorView, index : Int) -> TensorView {
  TensorView::{ tensor: self.tensor, index: [..self.index, Index(index)] }
}

pub fn TensorView::op_as_view(
  self : TensorView,
  start : Int,
  end : Int
) -> TensorView {
  TensorView::{ tensor: self.tensor, index: [..self.index, Range(start, end)] }
}

pub fn TensorView::output(self : TensorView, logger : Logger) -> Unit {
  self.tensor._output(logger, self.index, [])
}

pub fn TensorView::to_string(self : TensorView) -> String {
  Show::to_string(self.tensor.value)
}
