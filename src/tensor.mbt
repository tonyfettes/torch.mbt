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
  Get(Tensor, FixedArray[Int])
  Cat(Array[Tensor], FixedArray[(Int, Int)])
  Any(~ref : () -> Unit, ~propagate : () -> Unit)
} derive(Show)

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
  self.value[offset]
}

pub fn Tensor::set(self : Tensor, index : Array[Int], value : Double) -> Unit {
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
  self.value[offset] = value
}

fn Tensor::_output(
  self : Tensor,
  logger : Logger,
  range : Array[TensorIndex],
  index : Array[Int]
) -> Unit {
  if index.length() == self.shape.length() {
    logger.write_string(self.get(index).to_string())
    return
  }
  let dimension = index.length()
  fn output_range(start : Int, end : Int) {
    logger.write_string("[")
    index.push(0)
    for i = start; i < end; i = i + 1 {
      index[dimension] = i
      self._output(logger, range, index)
      if i < end - 1 {
        logger.write_string(", ")
      }
    }
    index.unsafe_pop() |> ignore
    logger.write_string("]")
  }

  if dimension >= range.length() {
    output_range(0, self.shape[dimension])
  } else {
    match range[dimension] {
      Index(range_index) => {
        index.push(range_index)
        self._output(logger, range, index)
        index.unsafe_pop() |> ignore
      }
      Slice(start, end) => output_range(start, end)
    }
  }
}

pub fn Tensor::output(self : Tensor, logger : Logger) -> Unit {
  self._output(logger, [], [])
}

pub fn Tensor::to_string(self : Tensor) -> String {
  Show::to_string(self.value)
}

pub fn Tensor::op_get(self : Tensor, index : Int) -> TensorView {
  if self.shape.length() == 0 {
    abort("Tensor is a scalar")
  }
  TensorView::{ tensor: self, path: [Index(index)] }
}

pub fn Tensor::op_set[X : Value](self : Tensor, index : Int, other : X) -> Unit {
  println("Tensor::op_set: \{self} \{index} \{other}")
  println("Tensor::op_set: self.size = \{self.size}")
  let other = other.to_tensor()
  let offset = index * self.size[0]
  for i = 0; i < self.size[0]; i = i + 1 {
    println("self.value[\{offset + i}] = other.value[\{i}]")
    self.value[offset + i] = other.value[i]
  }
}

test "Tensor::op_set" {
  let tensor = Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  tensor[1] = [7.0, 8.0, 9.0]
  inspect!(tensor, content="[[1, 2, 3], [7, 8, 9]]")
  tensor[0] = [0.0, 1.0, 2.0]
  inspect!(tensor, content="[[0, 1, 2], [7, 8, 9]]")
}

pub fn Tensor::length(self : Tensor) -> Int {
  self.shape[0]
}

pub fn Tensor::op_as_view(
  self : Tensor,
  ~start : Int,
  ~end? : Int
) -> TensorView {
  if self.shape.length() == 0 {
    abort("Tensor is a scalar")
  }
  let end = match end {
    None => self.shape[0]
    Some(end) => end
  }
  let path = [TensorIndex::Slice(start, end)]
  TensorView::{ tensor: self, path }
}

test "panic Tensor::op_as_view - Scalar" {
  let tensor = Tensor::new(1.0)
  tensor[0:1] |> ignore
}

test "Tensor::op_as_view - Vector" {
  let tensor = Tensor::new([0.0, 1.0, 2.0, 3.0])
  inspect!(tensor[1:3], content="[1, 2]")
  inspect!(tensor[1:], content="[1, 2, 3]")
  inspect!(tensor[:3], content="[0, 1, 2]")
}

test "Tensor::op_as_view - Matrix" {
  let tensor = Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  inspect!(tensor[1:2], content="[[3, 4, 5]]")
  inspect!(tensor[1:], content="[[3, 4, 5]]")
  inspect!(tensor[:2], content="[[0, 1, 2], [3, 4, 5]]")
  inspect!(tensor[:], content="[[0, 1, 2], [3, 4, 5]]")
  inspect!(tensor[:][1], content="[1, 4]")
  inspect!(tensor[:][1:], content="[[1, 2], [4, 5]]")
  inspect!(tensor[:][:], content="[[0, 1, 2], [3, 4, 5]]")
}

pub fn Tensor::to_tensor(self : Tensor) -> Tensor {
  self
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
  let other = if other.shape.length() == 0 {
    Tensor::new(Array::make(self.value.length(), other.value[0]))
  } else {
    other
  }
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
  if self.size.length() == 0 {
    abort("Cannot sum a scalar")
  }
  let value : FixedArray[Double] = FixedArray::make(self.size[0], 0.0)
  for i = 0; i < self.shape[0]; i = i + 1 {
    for j = 0; j < self.size[0]; j = j + 1 {
      value[j] = value[j] + self.value[i * self.size[0] + j]
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
    value,
    shape,
    size,
    graph: Graph::Sum(self),
    ref: 0,
    grad: FixedArray::make(self.size[0], 0.0),
  }
}

test "Tensor::sum - Vector" {
  let tensor = Tensor::new([0.0, 1.0, 2.0, 3.0])
  let sum = tensor.sum()
  inspect!(sum.value, content="[6]")
  inspect!(sum.shape, content="[]")
  inspect!(sum.size, content="[]")
  inspect!(sum.graph, content="Sum([0, 1, 2, 3])")
  inspect!(sum.ref, content="0")
  inspect!(sum.grad, content="[0]")
  inspect!(sum, content="6")
}

test "Tensor::sum - Matrix" {
  let tensor = Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let sum = tensor.sum()
  inspect!(sum.value, content="[3, 5, 7]")
  inspect!(sum.shape, content="[3]")
  inspect!(sum.size, content="[1]")
  inspect!(sum.graph, content="Sum([[0, 1, 2], [3, 4, 5]])")
  inspect!(sum.ref, content="0")
  inspect!(sum.grad, content="[0, 0, 0]")
  inspect!(sum, content="[3, 5, 7]")
}

pub fn Tensor::dot(self : Tensor, other : Tensor) -> Tensor {
  if self.shape.length() != 1 || other.shape.length() != 1 {
    abort("Cannot dot product non-vectors")
  }
  Tensor::sum(self * other)
}

pub fn Tensor::matmul(self : Tensor, other : Tensor) -> Tensor {
  println("Tensor::matmul: \{self.shape} \{other.shape}")
  if self.shape.length() != 2 || other.shape.length() != 2 {
    abort("Cannot matrix multiply non-matrices")
  }
  let output : FixedArray[Double] = FixedArray::make(
    self.shape[0] * other.shape[1],
    0.0,
  )
  for i = 0; i < self.shape[0]; i = i + 1 {
    for j = 0; j < other.shape[1]; j = j + 1 {
      for k = 0; k < self.shape[1]; k = k + 1 {
        output[i * other.shape[1] + j] += self.value[i * self.shape[1] + k] *
          other.value[k * other.shape[1] + j]
      }
    }
  }
  Tensor::{
    value: output,
    shape: FixedArray::from_array([self.shape[0], other.shape[1]]),
    size: FixedArray::from_array([other.shape[1], 1]),
    graph: Graph::MatMul(self, other),
    ref: 0,
    grad: FixedArray::make(self.shape[0] * other.shape[1], 0.0),
  }
}

test "Tensor::matmul" {
  let a = Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let b = Tensor::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
  let c = a.matmul(b)
  inspect!(c.value, content="[58, 64, 139, 154]")
  inspect!(c.shape, content="[2, 2]")
  inspect!(c.size, content="[2, 1]")
  inspect!(
    c.graph,
    content="MatMul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])",
  )
  inspect!(c.ref, content="0")
  inspect!(c.grad, content="[0, 0, 0, 0]")
  inspect!(c, content="[[58, 64], [139, 154]]")
}

pub trait Value: Show {
  to_tensor(Self) -> Tensor
}

pub trait Default {
  default() -> Self
}

pub impl Value for Double with to_tensor(self : Double) -> Tensor {
  Tensor::{
    value: [self],
    shape: [],
    size: [],
    graph: Graph::Val,
    ref: 0,
    grad: [0.0],
  }
}

pub impl Default for Double with default() -> Double { 0.0 }

pub fn Tensor::new[X : Value](self : X) -> Tensor {
  self.to_tensor()
}

pub fn tensor[X : Value](x : X) -> Tensor {
  x.to_tensor()
}

pub impl[X : Value + Default] Value for Array[X] with to_tensor(self : Array[X]) -> Tensor {
  let tensors = self.map(fn { x => x.to_tensor() })
  let shape = [self.length()]
  if self.length() == 0 {
    for s in X::default().to_tensor().shape {
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

pub impl[X] Default for Array[X] with default() -> Array[X] { [] }

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

pub enum TensorIndex {
  Index(Int)
  Slice(Int, Int)
}

pub struct TensorView {
  tensor : Tensor
  path : Array[TensorIndex]
}

pub fn TensorView::output(self : TensorView, logger : Logger) -> Unit {
  self.tensor._output(logger, self.path, [])
}

pub fn TensorView::op_get(self : TensorView, index : Int) -> TensorView {
  TensorView::{ tensor: self.tensor, path: [..self.path, Index(index)] }
}

pub fn TensorView::op_set(
  self : TensorView,
  index : Int,
  other : Value
) -> Unit {
  let other = other.to_tensor()
  fn assign(
    path : ArrayView[TensorIndex],
    self_indicies : Array[Int],
    other_indicies : Array[Int]
  ) -> Unit {
    if self_indicies.length() == self.tensor.shape.length() {
      let mut self_offset = 0
      for i = 0; i < self_indicies.length(); i = i + 1 {
        self_offset += self_indicies[i] * self.tensor.size[i]
      }
      let mut other_offset = 0
      for i = 0; i < other_indicies.length(); i = i + 1 {
        other_offset += other_indicies[i] * other.size[i]
      }
      self.tensor.value[self_offset] = other.value[other_offset]
      return
    }
    match path {
      [Index(index), .. as path] =>
        assign(path, [..self_indicies, index], other_indicies)
      [Slice(start, end), .. as path] =>
        for i = start; i < end; i = i + 1 {
          assign(path, [..self_indicies, i], [..other_indicies, i - start])
        }
      [] =>
        for i = 0; i < self.tensor.shape[self_indicies.length()]; i = i + 1 {
          assign(path, [..self_indicies, i], [..other_indicies, i])
        }
    }
  }

  assign([..self.path, Index(index)][:], [], [])
}

test "TensorView::op_set" {
  let tensor = Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  tensor[:][1] = [7.0, 8.0]
  inspect!(tensor, content="[[1, 7, 3], [4, 8, 6]]")
  tensor[:][0] = [0.0, 1.0]
  inspect!(tensor, content="[[0, 7, 3], [1, 8, 6]]")
  tensor[:][2] = [2.0, 3.0]
  inspect!(tensor, content="[[0, 7, 2], [1, 8, 3]]")
  tensor[0][0] = 9.0
  inspect!(tensor, content="[[9, 7, 2], [1, 8, 3]]")
}

pub fn TensorView::op_as_view(
  self : TensorView,
  ~start : Int,
  ~end? : Int
) -> TensorView {
  let end = match end {
    None => self.tensor.shape[self.path.length()]
    Some(end) => end
  }
  TensorView::{ tensor: self.tensor, path: [..self.path, Slice(start, end)] }
}

pub fn TensorView::op_add(self : TensorView, other : TensorView) -> Tensor {
  self.to_tensor() + other.to_tensor()
}

pub fn TensorView::op_sub(self : TensorView, other : TensorView) -> Tensor {
  self.to_tensor() + other.to_tensor()
}

pub fn TensorView::op_mul(self : TensorView, other : TensorView) -> Tensor {
  self.to_tensor() * other.to_tensor()
}

pub fn TensorView::log(self : TensorView) -> Tensor {
  self.to_tensor().log()
}

pub fn TensorView::to_tensor(self : TensorView) -> Tensor {
  let shape = []
  let value = []
  let mapping : Array[Int] = []
  fn build(path : ArrayView[TensorIndex], indicies : Array[Int]) -> Unit {
    if indicies.length() == self.tensor.shape.length() {
      let mut offset = 0
      for i = 0; i < indicies.length(); i = i + 1 {
        offset += indicies[i] * self.tensor.size[i]
      }
      value.push(self.tensor.value[offset])
      mapping.push(offset)
      return
    }
    match path {
      [Index(index), .. as path] => build(path, [..indicies, index])
      [Slice(start, stop), .. as path] => {
        shape.push(stop - start)
        for i = start; i < stop; i = i + 1 {
          build(path, [..indicies, i])
        }
      }
      [] => {
        shape.push(self.tensor.shape[indicies.length()])
        for i = 0; i < self.tensor.shape[indicies.length()]; i = i + 1 {
          build(path, [..indicies, i])
        }
      }
    }
  }

  build(self.path[:], [])
  let size = [1]
  for i = shape.length() - 1; i > 0; i = i - 1 {
    size.push(shape[i] * size[size.length() - 1])
  }
  size.rev_inplace()
  Tensor::{
    value: FixedArray::from_array(value),
    shape: FixedArray::from_array(shape),
    size: FixedArray::from_array(size),
    graph: Graph::Get(self.tensor, FixedArray::from_array(mapping)),
    ref: 0,
    grad: FixedArray::make(value.length(), 0.0),
  }
}

test "TensorView::to_tensor" {
  let tensor = Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let view = tensor[1]
  let tensor = view.to_tensor()
  inspect!(tensor.value, content="[3, 4, 5]")
  inspect!(tensor.shape, content="[3]")
  inspect!(tensor.size, content="[1]")
  inspect!(tensor.graph, content="Get([[0, 1, 2], [3, 4, 5]], [3, 4, 5])")
  inspect!(tensor.ref, content="0")
  inspect!(tensor.grad, content="[0, 0, 0]")
  inspect!(tensor, content="[3, 4, 5]")
}

fn Tensor::ref(self : Tensor) -> Unit {
  self.ref += 1
  if self.ref > 1 {
    return
  }
  match self.graph {
    Val => ()
    Var => ()
    Add(a, b) => {
      a.ref()
      b.ref()
    }
    Sub(a, b) => {
      a.ref()
      b.ref()
    }
    Mul(a, b) => {
      a.ref()
      b.ref()
    }
    Div(a, b) => {
      a.ref()
      b.ref()
    }
    Neg(x) => x.ref()
    Exp(x) => x.ref()
    Log(x) => x.ref()
    Sum(x) => x.ref()
    Pow(x, _) => x.ref()
    MatMul(a, b) => {
      a.ref()
      b.ref()
    }
    Cat(xs, _) =>
      for x in xs {
        x.ref()
      }
    Get(x, _) => x.ref()
    Any(~ref, ..) => ref()
  }
}

fn power(base : Double, exponent : Int) -> Double {
  if exponent == 0 {
    return 1.0
  }
  if exponent == 1 {
    return base
  }
  let mut result = 1.0
  let mut base = base
  let mut exponent = exponent
  while exponent > 0 {
    if exponent % 2 == 1 {
      result = result * base
    }
    base = base * base
    exponent = exponent / 2
  }
  result
}

test "power" {
  inspect!(power(2.0, 0), content="1")
  inspect!(power(2.0, 1), content="2")
  inspect!(power(0.0, 5), content="0")
  inspect!(power(1.0, 10), content="1")
  inspect!(power(3.0, 3), content="27")
  inspect!(power(4.0, 2), content="16")
  inspect!(power(5.0, 3), content="125")
  inspect!(power(2.0, 10), content="1024")
  inspect!(power(1.5, 3), content="3.375")
  inspect!(power(2.5, 2), content="6.25")
  inspect!(power(2.0, 100), content="1.2676506002282294e+30")
}

pub fn Tensor::pow(self : Tensor, exponent : Int) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = power(self.value[i], exponent)
  }
  Tensor::{
    value: output,
    shape: self.shape,
    size: self.size,
    graph: Graph::Pow(self, exponent),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

fn Tensor::propagate(self : Tensor) -> Unit {
  self.ref -= 1
  if self.ref != 0 {
    return
  }
  match self.graph {
    Val => return
    Var => return
    Add(a, b) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        a.grad[i] += self.grad[i]
        b.grad[i] += self.grad[i]
      }
      a.propagate()
      b.propagate()
    }
    Sub(a, b) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        a.grad[i] -= self.grad[i]
        b.grad[i] -= self.grad[i]
      }
      a.propagate()
      b.propagate()
    }
    Mul(a, b) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        a.grad[i] += self.grad[i] * b.value[i]
        b.grad[i] += self.grad[i] * a.value[i]
      }
      a.propagate()
      b.propagate()
    }
    Div(a, b) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        a.grad[i] += self.grad[i] / b.value[i]
        b.grad[i] += -self.grad[i] * a.value[i] / b.value[i] / b.value[i]
      }
      a.propagate()
      b.propagate()
    }
    Neg(a) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        a.grad[i] -= self.grad[i]
      }
      a.propagate()
    }
    Exp(a) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        a.grad[i] += self.grad[i] * self.value[i]
      }
      a.propagate()
    }
    Log(a) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        a.grad[i] += self.grad[i] / a.value[i]
      }
      a.propagate()
    }
    Sum(x) => {
      for i = 0; i < x.shape[0]; i = i + 1 {
        for j = 0; j < x.size[0]; j = j + 1 {
          x.grad[i * x.size[0] + j] += self.grad[j]
        }
      }
      x.propagate()
    }
    Pow(a, b) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        a.grad[i] = self.grad[i] * b.to_double() * power(a.value[i], b - 1)
      }
      a.propagate()
    }
    MatMul(a, b) => {
      for i = 0; i < a.shape[0]; i = i + 1 {
        for j = 0; j < b.shape[1]; j = j + 1 {
          for k = 0; k < a.shape[1]; k = k + 1 {
            a.grad[i * a.size[0] + k] += self.grad[i * a.shape[0] + j] *
              b.value[k * b.size[0] + j]
            b.grad[k * b.size[0] + j] += self.grad[i * a.shape[0] + j] *
              a.value[i * a.size[0] + k]
          }
        }
      }
      a.propagate()
      b.propagate()
    }
    Get(x, m) => {
      for i = 0; i < m.length(); i = i + 1 {
        x.grad[m[i]] += self.grad[i]
      }
      x.propagate()
    }
    Cat(xs, m) =>
      for i = 0; i < self.grad.length(); i = i + 1 {
        let (x, m) = m[i]
        xs[x].grad[m] += self.grad[i]
        xs[x].propagate()
      }
    Any(~propagate, ..) => propagate()
  }
}

fn Tensor::check(self : Tensor) -> Unit {
  if self.ref != 0 {
    abort("ref != 0")
  }
  match self.graph {
    Val => ()
    Var => ()
    Add(x, y) => {
      x.check()
      y.check()
    }
    Sub(x, y) => {
      x.check()
      y.check()
    }
    Mul(x, y) => {
      x.check()
      y.check()
    }
    Div(x, y) => {
      x.check()
      y.check()
    }
    Neg(x) => x.check()
    Exp(x) => x.check()
    Log(x) => x.check()
    Sum(x) => x.check()
    Pow(x, _) => x.check()
    MatMul(x, y) => {
      x.check()
      y.check()
    }
    Get(a, _) => a.check()
    Cat(xs, _) =>
      for x in xs {
        x.check()
      }
    Any(..) => ()
  }
}

pub fn Tensor::backward(self : Tensor) -> Unit {
  self.ref()
  for i = 0; i < self.grad.length(); i = i + 1 {
    self.grad[i] += 1.0
  }
  self.propagate()
  self.check()
}

test "Tensor::backward - Add" {
  let a = Tensor::new(1.0)
  let b = Tensor::new(2.0)
  let c = a + b
  c.backward()
  inspect!(a.grad, content="[1]")
  inspect!(b.grad, content="[1]")
}

test "Tensor::backward - Mul" {
  let a = Tensor::new(3.0)
  let b = Tensor::new(4.0)
  let c = a * b
  c.backward()
  inspect!(a.grad, content="[4]")
  inspect!(b.grad, content="[3]")
}

test "Tensor::bacward - MatMul" {
  let a = [
    [Tensor::new(1.0), Tensor::new(2.0), Tensor::new(3.0)],
    [Tensor::new(4.0), Tensor::new(5.0), Tensor::new(6.0)],
  ]
  let b = [
    [Tensor::new(7.0), Tensor::new(8.0)],
    [Tensor::new(9.0), Tensor::new(10.0)],
    [Tensor::new(11.0), Tensor::new(12.0)],
  ]
  let c = []
  for i = 0; i < a.length(); i = i + 1 {
    let row = []
    for j = 0; j < b[0].length(); j = j + 1 {
      let mut sum = Tensor::new(0.0)
      for k = 0; k < a[0].length(); k = k + 1 {
        sum = sum + a[i][k] * b[k][j]
      }
      row.push(sum)
    }
    c.push(row)
  }
  inspect!(c, content="[[58, 64], [139, 154]]")
  for row in c {
    for col in row {
      col.backward()
    }
  }
  inspect!(a[0][0].grad, content="[15]")
  inspect!(a[0][1].grad, content="[19]")
  inspect!(a[0][2].grad, content="[23]")
  inspect!(a[1][0].grad, content="[15]")
  inspect!(a[1][1].grad, content="[19]")
  inspect!(a[1][2].grad, content="[23]")
  inspect!(b[0][0].grad, content="[5]")
  inspect!(b[0][1].grad, content="[5]")
  inspect!(b[1][0].grad, content="[7]")
  inspect!(b[1][1].grad, content="[7]")
  inspect!(b[2][0].grad, content="[9]")
  inspect!(b[2][1].grad, content="[9]")
  let x = Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let y = Tensor::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
  let z = x.matmul(y)
  inspect!(z, content="[[58, 64], [139, 154]]")
  z.backward()
  inspect!(x.grad, content="[15, 19, 23, 15, 19, 23]")
  inspect!(y.grad, content="[5, 5, 7, 7, 9, 9]")
}

pub fn Tensor::reshape(self : Tensor, shape : Array[Int]) -> Tensor {
  let size = [1]
  for i = shape.length() - 1; i > 0; i = i - 1 {
    size.push(shape[i] * size[size.length() - 1])
  }
  Tensor::{
    value: self.value.copy(),
    shape: FixedArray::from_array(shape),
    size: FixedArray::from_array(size),
    ref: 0,
    graph: Get(self, FixedArray::makei(self.value.length(), fn { i => i })),
    grad: FixedArray::make(self.value.length(), 0.0),
  }
}

pub fn Tensor::reLU(self : Tensor) -> Tensor {
  let x = self
  let self = Tensor::{
    value: self.value.map(fn { x => if x > 0.0 { x } else { 0.0 } }),
    shape: self.shape,
    size: self.size,
    graph: Graph::Any(
      ref=fn() {
        self.ref += 1
        if self.ref > 1 {
          return
        }
        self.ref()
      },
      propagate=fn() {
        for i = 0; i < self.grad.length(); i = i + 1 {
          x.grad[i] += self.grad[i] * if self.value[i] > 0.0 { 1.0 } else { 0.0 }
        }
        x.propagate()
      },
    ),
    ref: 0,
    grad: FixedArray::make(self.value.length(), 0.0),
  }
  self
}
