pub struct Tensor {
  value : FixedArray[Double]
  shape : FixedArray[Int]
  block : FixedArray[Int]
  mut graph : Graph
  mut ref : Int
  grad : FixedArray[Double]
}

pub fn Tensor::shape(self : Tensor) -> FixedArray[Int] {
  self.shape
}

pub let empty : Tensor = Tensor::{
  value: [],
  shape: [],
  block: [],
  graph: Graph::Val,
  ref: 0,
  grad: [],
}

pub fn Tensor::to_json(self : Tensor) -> Json {
  fn build(dimension : Int, offset : Int) -> Json {
    if dimension >= self.shape.length() {
      return Json::Number(self.value[offset])
    }
    let elements = Array::new(capacity=self.shape[dimension])
    for i = 0; i < self.shape[dimension]; i = i + 1 {
      elements.push(build(dimension + 1, offset + i * self.block[dimension]))
    }
    Json::Array(elements)
  }

  build(0, 0)
}

test "Tensor::to_json" {
  let tensor = tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}

pub fn Tensor::from_json(
  json : Json,
  path : @json.JsonPath
) -> Tensor!@json.JsonDecodeError {
  match json {
    Json::Number(x) => Tensor::new(x)
    Json::Array(xs) => {
      let value : Array[Tensor] = Array::new(capacity=xs.length())
      for i, x in xs {
        value.push(Tensor::from_json!(x, @json.add_index(path, i)))
      }
      Tensor::stack(FixedArray::from_array(value))
    }
    _ => raise @json.JsonDecodeError((path, "Expected an array"))
  }
}

test "Tensor::from_json" {
  let json : Json = [[0, 1, 2], [3, 4, 5]]
  let tensor : Tensor = json |> @json.from_json!()
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
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
  Sum(Tensor, FixedArray[Array[Int]])
  Pow(Tensor, Int)
  MatMul(Tensor, Tensor)
  Get(Tensor, FixedArray[Int])
  Cat(FixedArray[Tensor], FixedArray[(Int, Int)])
  Nop(Tensor)
  ReLU(Tensor)
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
    offset += index[i] * self.block[i]
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
    offset += index[i] * self.block[i]
  }
  self.value[offset] = value
}

pub fn Tensor::output(self : Tensor, logger : Logger) -> Unit {
  let shape = self.shape
  let dimension = shape.length()
  let value = self.value
  fn build(d : Int, p : Int) {
    if d == dimension {
      Show::output(value[p], logger)
      return
    }
    logger.write_char('[')
    for i = 0; i < shape[d]; i = i + 1 {
      build(d + 1, p + i * self.block[d])
      if i < shape[d] - 1 {
        logger.write_string(", ")
      }
    }
    logger.write_char(']')
  }

  build(0, 0)
}

pub fn Tensor::to_string(self : Tensor) -> String {
  Show::to_string(self.value)
}

pub fn Tensor::op_get(self : Tensor, index : Int) -> TensorView {
  if self.shape.length() == 0 {
    abort("Tensor is a scalar")
  }
  if index >= self.shape[0] {
    abort("Tensor element access \{index} out of bounds [0, \{self.shape[0]}")
  }
  TensorView::{ tensor: self, index: [Index(index)] }
}

pub fn Tensor::op_set[X : TensorValue](
  self : Tensor,
  index : Int,
  other : X
) -> Unit {
  if not(no_grad.val) {
    match self.graph {
      Val => ()
      _ =>
        abort(
          "a view of a leaf Variable that requires grad is being used in an in-place operation.",
        )
    }
  }
  let other = other.to_tensor(requires_grad=false)
  let offset = index * self.block[0]
  for i = 0; i < self.block[0]; i = i + 1 {
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
    Some(end) => {
      if end > self.shape[0] {
        abort(
          "Tensor slice [\{start}, \{end}) out of bounds: [0, \{self.shape[0]})",
        )
      }
      end
    }
  }
  let index = [TensorIndex::Slice(start, end)]
  TensorView::{ tensor: self, index }
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

pub impl TensorLike for Tensor with to_tensor(self : Tensor) -> Tensor { self }

fn Tensor::default() -> Tensor {
  abort("Unable to construct a tensor with default shape")
}

pub fn Tensor::op_sub(self : Tensor, other : Tensor) -> Tensor {
  let (self, other) = Tensor::broadcast(self, other)
  let length = self.value.length()
  let value : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    value[i] = self.value[i] - other.value[i]
  }
  Tensor::{
    value,
    shape: self.shape,
    block: self.block,
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
    block: self.block,
    graph: Graph::Neg(self),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::op_div(self : Tensor, other : Tensor) -> Tensor {
  let (self, other) = Tensor::broadcast(self, other)
  let length = self.value.length()
  let output : FixedArray[Double] = FixedArray::make(length, 0.0)
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = self.value[i] / other.value[i]
  }
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
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
    block: self.block,
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
    block: self.block,
    graph: Graph::Log(self),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

pub fn Tensor::sum(self : Tensor, ~dim : @shape.Shape = ()) -> Tensor {
  let dim = dim.to_fixed_array()
  let dim = if dim.length() == 0 {
    FixedArray::makei(self.shape.length(), fn(i) { i })
  } else {
    dim
  }
  let do_sum = FixedArray::make(self.shape.length(), false)
  for d in dim {
    do_sum[d] = true
  }
  let shape = []
  for d in 0..<self.shape.length() {
    if not(do_sum[d]) {
      shape.push(self.shape[d])
    }
  }
  let shape = FixedArray::from_array(shape)
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value : FixedArray[Double] = FixedArray::make(total, 0.0)
  let index : FixedArray[Array[Int]] = FixedArray::makei(total, fn(__) { [] })
  fn build(d : Int, sum_pos : Int, val_pos : Int) -> Unit {
    if d == self.shape.length() {
      value[sum_pos] += self.value[val_pos]
      index[sum_pos].push(val_pos)
      return
    }
    if do_sum[d] {
      for i in 0..<self.shape[d] {
        build(d + 1, sum_pos, val_pos + i * self.block[d])
      }
    } else {
      for i in 0..<self.shape[d] {
        build(d + 1, sum_pos + i * self.block[d], val_pos + i * self.block[d])
      }
    }
  }

  build(0, 0, 0)
  Tensor::{
    value,
    shape,
    block,
    graph: Graph::Sum(self, index),
    ref: 0,
    grad: FixedArray::make(total, 0.0),
  }
}

test "Tensor::sum - scalar" {
  let tensor = Tensor::new(1.0)
  let sum = tensor.sum()
  inspect!(sum.value, content="[1]")
  inspect!(sum.shape, content="[]")
  inspect!(sum.block, content="[]")
  inspect!(sum.graph, content="Sum(1, [[0]])")
  inspect!(sum.ref, content="0")
  inspect!(sum.grad, content="[0]")
  inspect!(sum, content="1")
}

test "Tensor::sum - Vector" {
  let tensor = Tensor::new([0.0, 1.0, 2.0, 3.0])
  let sum = tensor.sum()
  inspect!(sum.value, content="[6]")
  inspect!(sum.shape, content="[]")
  inspect!(sum.block, content="[]")
  inspect!(sum.graph, content="Sum([0, 1, 2, 3], [[0, 1, 2, 3]])")
  inspect!(sum.ref, content="0")
  inspect!(sum.grad, content="[0]")
  inspect!(sum, content="6")
}

test "Tensor::sum - Matrix" {
  let tensor = Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let sum = tensor.sum(dim=0)
  inspect!(sum.value, content="[3, 5, 7]")
  inspect!(sum.shape, content="[3]")
  inspect!(sum.block, content="[1]")
  inspect!(
    sum.graph,
    content="Sum([[0, 1, 2], [3, 4, 5]], [[0, 3], [1, 4], [2, 5]])",
  )
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

pub fn Tensor::mean(self : Tensor) -> Tensor {
  self.sum() / tensor(self.value.length().to_double())
}

pub trait TensorValue: Show {
  to_tensor(Self, ~requires_grad : Bool) -> Tensor
}

pub trait TensorLike {
  to_tensor(Self) -> Tensor
}

pub trait Default {
  default() -> Self
}

pub impl TensorValue for Double with to_tensor(
  self : Double,
  ~requires_grad : Bool
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
    ref: 0,
    grad: [0.0],
  }
}

pub impl Default for Double with default() -> Double { 0.0 }

pub fn Tensor::new[X : TensorValue](
  self : X,
  ~requires_grad : Bool = false
) -> Tensor {
  self.to_tensor(~requires_grad)
}

pub fn tensor[X : TensorValue](x : X, ~requires_grad : Bool = false) -> Tensor {
  x.to_tensor(~requires_grad)
}

pub impl[X : TensorValue + Default] TensorValue for FixedArray[X] with to_tensor(
  self : FixedArray[X],
  ~requires_grad : Bool
) -> Tensor {
  let tensors = self.map(fn { x => x.to_tensor(~requires_grad) })
  let element = if self.length() == 0 {
    X::default().to_tensor(~requires_grad)
  } else {
    tensors[0]
  }
  let shape = FixedArray::make(element.shape.length() + 1, 0)
  shape[0] = self.length()
  for i, s in element.shape {
    shape[i + 1] = s
  }
  let block = compute_block_size(shape)
  let value = FixedArray::make(shape[0] * block[0], 0.0)
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
    ref: 0,
    grad: FixedArray::make(value.length(), 0.0),
  }
}

pub impl[X : TensorValue + Default] TensorValue for Array[X] with to_tensor(
  self : Array[X],
  ~requires_grad : Bool
) -> Tensor {
  let tensors = self.map(fn { x => x.to_tensor(~requires_grad) })
  let element = if self.length() == 0 {
    X::default().to_tensor(~requires_grad)
  } else {
    tensors[0]
  }
  let shape = FixedArray::make(element.shape.length() + 1, 0)
  shape[0] = self.length()
  for i, s in element.shape {
    shape[i + 1] = s
  }
  let block = compute_block_size(shape)
  let value = FixedArray::make(shape[0] * block[0], 0.0)
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
  inspect!(tensor.block, content="[]")
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
  inspect!(tensor.block, content="[1]")
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
  inspect!(tensor.block, content="[3, 1]")
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
  inspect!(tensor.block, content="[12, 4, 1]")
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

let no_grad : Ref[Bool] = Ref::new(false)

pub fn with_no_grad[X](f : () -> X) -> X {
  no_grad.protect(true, f)
}

pub fn with_no_grad_error[X](f : () -> X!) -> X! {
  let old = no_grad.val
  no_grad.val = true
  let r = f!()
  no_grad.val = old
  r
}

fn compute_block_size(shape : FixedArray[Int]) -> FixedArray[Int] {
  let mut dimension = shape.length()
  let block = FixedArray::make(dimension, 1)
  dimension -= 1
  while dimension > 0 {
    block[dimension - 1] = block[dimension] * shape[dimension]
    dimension -= 1
  }
  block
}

test "compute_block_size" {
  inspect!(compute_block_size([2, 3, 4]), content="[12, 4, 1]")
}

fn Tensor::ref(self : Tensor) -> Unit {
  if self.ref == -1 {
    self.ref = 0
  }
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
    Sum(x, _) => x.ref()
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
    Nop(x) => x.ref()
    ReLU(x) => x.ref()
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
    block: self.block,
    graph: Graph::Pow(self, exponent),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}

fn Tensor::propagate(self : Tensor) -> Unit {
  if self.ref == 0 {
    abort("Propagate through a dead tensor: \{self}")
  }
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
        a.grad[i] = self.grad[i]
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
      if b.shape.length() == 0 {
        for i = 0; i < self.grad.length(); i = i + 1 {
          a.grad[i] += self.grad[i] / b.value[0]
          b.grad[0] += -self.grad[i] * a.value[i] / b.value[0] / b.value[0]
        }
      } else {
        for i = 0; i < self.grad.length(); i = i + 1 {
          a.grad[i] += self.grad[i] / b.value[i]
          b.grad[i] += -self.grad[i] * a.value[i] / b.value[i] / b.value[i]
        }
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
    Sum(x, m) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        for j = 0; j < m[i].length(); j = j + 1 {
          x.grad[m[i][j]] += self.grad[i]
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
            a.grad[i * a.block[0] + k] += self.grad[i * a.shape[0] + j] *
              b.value[k * b.block[0] + j]
            b.grad[k * b.block[0] + j] += self.grad[i * a.shape[0] + j] *
              a.value[i * a.block[0] + k]
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
    Cat(xs, m) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        let (x, m) = m[i]
        xs[x].grad[m] += self.grad[i]
      }
      for x in xs {
        x.propagate()
      }
    }
    Nop(x) => x.propagate()
    ReLU(x) => {
      for i = 0; i < self.grad.length(); i = i + 1 {
        x.grad[i] += self.grad[i] *
          (if self.value[i] > 0.0 { 1.0 } else { 0.0 })
      }
      x.propagate()
    }
  }
}

fn Tensor::check(self : Tensor) -> Unit {
  if self.ref == -1 {
    return
  }
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
    Sum(x, _) => x.check()
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
    Nop(x) => x.check()
    ReLU(x) => x.check()
  }
  self.ref = -1
}

pub fn Tensor::backward(self : Tensor) -> Unit {
  self.ref()
  for i = 0; i < self.grad.length(); i = i + 1 {
    self.grad[i] += 1.0
  }
  self.propagate()
  if @debug.debug {
    self.check()
  }
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
    return tensors[0].upgrade()
  }
  for i in 1..<tensors.length() {
    if tensors[i].shape != tensors[0].shape {
      abort(
        "Tensor at \{i} has shape \{tensors[i].shape}, which is not the same as \{tensors[0].shape}",
      )
    }
  }
  let shape = FixedArray::make(tensors[0].shape.length() + 1, 0)
  shape[0] = tensors.length()
  for i, s in tensors[0].shape {
    shape[i + 1] = s
  }
  let block = compute_block_size(shape)
  let value = FixedArray::make(shape[0] * block[0], 0.0)
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
    grad: FixedArray::make(value.length(), 0.0),
  }
}

pub fn Tensor::stack[X : TensorLike](tensors : FixedArray[X]) -> Tensor {
  TensorLike::to_tensor(tensors)
}

test "Tensor::stack" {
  let input : FixedArray[_] = [
    [[tensor(0.0), tensor(1.0), tensor(2.0)]],
    [[tensor(3.0), tensor(4.0), tensor(5.0)]],
  ]
  let input = stack(input)
  inspect!(input, content="[[[0, 1, 2]], [[3, 4, 5]]]")
  inspect!(
    input.graph,
    content="Cat([[[0, 1, 2]], [[3, 4, 5]]], [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])",
  )
}

pub fn Tensor::cat(tensors : FixedArray[Tensor], ~dim : Int = 0) -> Tensor {
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
  let value = Array::new(capacity=shape[0] * block[0])
  let index = Array::new(capacity=shape[0] * block[0])
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
    ref: 0,
    grad: FixedArray::make(value.length(), 0.0),
  }
}

pub fn Tensor::reLU(self : Tensor) -> Tensor {
  Tensor::{
    value: self.value.map(fn { x => if x > 0.0 { x } else { 0.0 } }),
    shape: self.shape,
    block: self.block,
    graph: Graph::ReLU(self),
    ref: 0,
    grad: FixedArray::make(self.value.length(), 0.0),
  }
}

pub fn Tensor::to_double(self : Tensor) -> Double {
  if self.shape.length() != 0 {
    abort("Tensor is not a scalar")
  }
  self.value[0]
}

test "Tensor::op_as_view" {
  let image = tensor(
    [
      [0.0, 1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0, 7.0],
      [8.0, 9.0, 10.0, 11.0],
      [12.0, 13.0, 14.0, 15.0],
    ],
  )
  inspect!(image[0:3][0:3], content="[[0, 1, 2], [4, 5, 6], [8, 9, 10]]")
  inspect!(
    image[0:3][0:3].to_tensor(),
    content="[[0, 1, 2], [4, 5, 6], [8, 9, 10]]",
  )
}

pub fn Tensor::permute(self : Tensor, permutation : Array[Int]) -> Tensor {
  if permutation.length() != self.shape.length() {
    abort("Permutation length does not match tensor rank")
  }
  let shape = FixedArray::makei(
    self.shape.length(),
    fn(i) { self.shape[permutation[i]] },
  )
  let value = Array::new(capacity=self.value.length())
  let index = Array::new(capacity=self.value.length())
  fn build(dimension : Int, offset : Int) -> Unit {
    if dimension == self.shape.length() {
      value.push(self.value[offset])
      index.push(offset)
      return
    }
    for i = 0; i < self.shape[permutation[dimension]]; i = i + 1 {
      build(dimension + 1, offset + i * self.block[permutation[dimension]])
    }
  }

  build(0, 0)
  Tensor::{
    value: FixedArray::from_array(value),
    shape,
    block: compute_block_size(shape),
    graph: Graph::Get(self, FixedArray::from_array(index)),
    ref: 0,
    grad: FixedArray::make(value.length(), 0.0),
  }
}

test "Tensor::permute" {
  let tensor = tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  inspect!(tensor.shape, content="[2, 3]")
  inspect!(tensor.value, content="[0, 1, 2, 3, 4, 5]")
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
  let permuted = tensor.permute([1, 0])
  inspect!(permuted.shape, content="[3, 2]")
  inspect!(permuted.value, content="[0, 3, 1, 4, 2, 5]")
  inspect!(permuted, content="[[0, 3], [1, 4], [2, 5]]")
}

pub fn Tensor::moveaxis[Shape : @shape.Shape](
  self : Tensor,
  source : Shape,
  destination : Shape
) -> Tensor {
  let source = source.to_fixed_array()
  let destination = destination.to_fixed_array()
  if source.length() != destination.length() {
    abort(
      "Source axes (\{source}) and destination axes (\{destination}) must have the same length",
    )
  }
  let permutation = Array::make(self.shape.length(), -1)
  let arranged = Array::make(self.shape.length(), false)
  for i = 0; i < source.length(); i = i + 1 {
    let source = if source[i] < 0 {
      source[i] + self.shape.length()
    } else {
      source[i]
    }
    let destination = if destination[i] < 0 {
      destination[i] + self.shape.length()
    } else {
      destination[i]
    }
    if permutation[destination] != -1 {
      abort(
        "Destination axis \{destination} is already assigned to \{permutation[destination]}",
      )
    }
    permutation[destination] = source
  }
  let mut dimension = 0
  for i = 0; i < permutation.length(); i = i + 1 {
    if permutation[i] != -1 {
      continue
    }
    while arranged[dimension] {
      dimension += 1
    }
    arranged[dimension] = true
    permutation[i] = dimension
  }
  self.permute(permutation)
}

test "Tensor::moveaxis" {
  let tensor = tensor(
    [
      [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
      [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
      [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
      [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
    ],
  )
  inspect!(tensor.shape, content="[4, 2, 3]")
  let moved = tensor.moveaxis(-1, 0)
  inspect!(moved.shape, content="[3, 4, 2]")
  inspect!(
    moved,
    content="[[[0, 0], [0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2], [2, 2]]]",
  )
}

pub fn Tensor::zeros[Shape : @shape.Shape](
  shape : Shape,
  ~requires_grad : Bool = false
) -> Tensor {
  let shape = shape.to_fixed_array()
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value = FixedArray::make(total, 0.0)
  Tensor::{
    shape,
    block,
    value,
    ref: 0,
    graph: if requires_grad {
      Graph::Var
    } else {
      Graph::Val
    },
    grad: FixedArray::make(total, 0.0),
  }
}

fn Tensor::ones[Shape : @shape.Shape](
  shape : Shape,
  ~requires_grad : Bool = false
) -> Tensor {
  let shape = shape.to_fixed_array()
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value = FixedArray::make(total, 1.0)
  Tensor::{
    shape,
    block,
    value,
    ref: 0,
    graph: if requires_grad {
      Graph::Var
    } else {
      Graph::Val
    },
    grad: FixedArray::make(total, 0.0),
  }
}

pub fn Tensor::upgrade(self : Tensor, ~dim : Int = 1) -> Tensor {
  let old_dim = self.shape.length()
  let shape = FixedArray::make(old_dim + dim, 1)
  for i = 0; i < old_dim; i = i + 1 {
    shape[i + dim] = self.shape[i]
  }
  self.reshape(shape)
}

pub fn Tensor::value(self : Tensor) -> FixedArray[Double] {
  self.value
}

pub fn Tensor::max(self : Tensor) -> Tensor {
  if self.value.length() == 0 {
    abort("Cannot take the maximum of an empty tensor")
  }
  let mut max_value = self.value[0]
  let mut max_index = 0
  for i in 1..<self.value.length() {
    if self.value[i] > max_value {
      max_value = self.value[i]
      max_index = i
    }
  }
  Tensor::{
    value: [max_value],
    shape: [],
    block: [],
    graph: Graph::Get(self, [max_index]),
    ref: 0,
    grad: [0.0],
  }
}

test "Tensor::max" {
  let x = tensor([1.0, 2.0, 3.0, 2.0, 1.0])
  let max = x.max()
  inspect!(max, content="3")
  max.backward()
  inspect!(x.grad, content="[0, 0, 1, 0, 0]")
}
