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
  println("offset: \{offset}")
  self.value[offset]
}

pub enum Index {
  Index(Int)
  Slice(Int, Int)
}

fn Tensor::_output(
  self : Tensor,
  logger : Logger,
  range : Array[Index],
  index : Array[Int]
) -> Unit {
  println("index: \{index}")
  println("shape: \{self.shape}")
  if index.length() == self.shape.length() {
    logger.write_string(self.get(index).to_string())
    return
  }
  let dimension = index.length()
  fn output_range(start : Int, end : Int) {
    logger.write_string("[")
    println("dimension: \{dimension}")
    index.push(0)
    for i = start; i < end; i = i + 1 {
      index[dimension] = i
      self._output(logger, range, index)
      if i < end - 1 {
        logger.write_string(", ")
      }
    }
    index.pop_exn() |> ignore
    logger.write_string("]")
  }

  if dimension >= range.length() {
    output_range(0, self.shape[dimension])
  } else {
    match range[dimension] {
      Index(range_index) => {
        index.push(range_index)
        self._output(logger, range, index)
        index.pop_exn() |> ignore
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

pub trait Value: Show + Default {
  tensor(Self) -> Tensor
}

trait Default {
  default() -> Self
}

impl Value for Double with tensor(self : Double) -> Tensor {
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

impl[X : Value + Default] Value for Array[X] with tensor(self : Array[X]) -> Tensor {
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

pub fn Tensor::new[X : Value](
  value : X,
  ~requires_grad : Bool = false
) -> Tensor {
  let tensor = value.tensor()
  if requires_grad {
    tensor.graph = Graph::Var
  }
  tensor
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

pub struct Slice {
  tensor : Tensor
  path : Array[Index]
}

pub fn Tensor::op_get(self : Tensor, index : Int) -> Slice {
  if self.shape.length() == 0 {
    abort("Tensor is a scalar")
  }
  Slice::{ tensor: self, path: [Index(index)] }
}

test "panic Tensor::op_get - Scalar" {
  let tensor = Tensor::new(1.0)
  tensor[0] |> ignore
}

test "Tensor::op_get - Vector" {
  let tensor = Tensor::new([0.0, 1.0, 2.0, 3.0])
  let slice = tensor[1]
  inspect!(slice, content="1")
}

test "Tensor::op_get - Matrix" {
  let tensor = Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let view = tensor[1]
  inspect!(view, content="[3, 4, 5]")
}

pub fn Tensor::length(self : Tensor) -> Int {
  self.shape[0]
}

pub fn Tensor::op_as_view(self : Tensor, ~start : Int, ~end? : Int) -> Slice {
  if self.shape.length() == 0 {
    abort("Tensor is a scalar")
  }
  let end = match end {
    None => self.shape[0]
    Some(end) => end
  }
  let path = [Index::Slice(start, end)]
  Slice::{ tensor: self, path }
}

pub fn Slice::output(self : Slice, logger : Logger) -> Unit {
  self.tensor._output(logger, self.path, [])
}

pub fn Slice::op_get(self : Slice, index : Int) -> Slice {
  Slice::{ tensor: self.tensor, path: [..self.path, Index(index)] }
}

pub fn Slice::op_as_view(self : Slice, ~start : Int, ~end? : Int) -> Slice {
  let end = match end {
    None => self.tensor.shape[self.path.length()]
    Some(end) => end
  }
  Slice::{ tensor: self.tensor, path: [..self.path, Slice(start, end)] }
}

pub fn Slice::tensor(self : Slice) -> Tensor {
  let shape = []
  let value = []
  let mapping : Array[Int] = []
  println("self.tensor.shape: \{self.tensor.shape}")
  println("self.tensor.size: \{self.tensor.size}")
  println("self.tensor.value: \{self.tensor.value}")
  fn build(path : ArrayView[Index], indicies : Array[Int]) -> Unit {
    println("shape: \{shape}")
    println("indicies: \{indicies}")
    if indicies.length() == self.tensor.shape.length() {
      let mut offset = 0
      for i = 0; i < indicies.length(); i = i + 1 {
        offset += indicies[i] * self.tensor.size[i]
      }
      println("offset: \{offset}")
      println("self.tensor.value[offset]: \{self.tensor.value[offset]}")
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
  println("shape: \{shape}")
  println("value: \{value}")
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

pub fn Tensor::tensor(self : Tensor) -> Tensor {
  self
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
