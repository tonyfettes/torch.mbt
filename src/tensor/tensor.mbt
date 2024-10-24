pub struct Tensor {
  value : FixedArray[Float]
  shape : FixedArray[Int]
  block : FixedArray[Int]
  mut graph : Graph
  mut ref : Int
  grad : FixedArray[Float]
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

pub fn Tensor::get(self : Tensor, index : Array[Int]) -> Float {
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

pub fn Tensor::set(self : Tensor, index : Array[Int], value : Float) -> Unit {
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

pub impl TensorLike for Tensor with to_tensor(self : Tensor) -> Tensor { self }

fn Tensor::default() -> Tensor {
  abort("Unable to construct a tensor with default shape")
}

pub fn Tensor::op_sub(self : Tensor, other : Tensor) -> Tensor {
  let (self, other) = Tensor::broadcast(self, other)
  let length = self.value.length()
  let value : FixedArray[Float] = FixedArray::make(length, 0.0)
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
  let output : FixedArray[Float] = FixedArray::make(length, 0.0)
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
  let output : FixedArray[Float] = FixedArray::make(length, 0.0)
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
  let output : FixedArray[Float] = FixedArray::make(length, 0.0)
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
  let output : FixedArray[Float] = FixedArray::make(length, 0.0)
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

pub fn Tensor::dot(self : Tensor, other : Tensor) -> Tensor {
  if self.shape.length() != 1 || other.shape.length() != 1 {
    abort("Cannot dot product non-vectors")
  }
  Tensor::sum(self * other)
}

pub fn Tensor::mean(self : Tensor) -> Tensor {
  self.sum() / tensor(self.value.length().to_double())
}

pub trait TensorLike {
  to_tensor(Self) -> Tensor
}

pub trait Default {
  default() -> Self
}

pub fn Tensor::new[X : TensorValue](
  self : X,
  ~requires_grad : Bool = false
) -> Tensor {
  self.to_tensor(~requires_grad)
}

pub fn tensor[X : TensorValue](x : X, ~requires_grad : Bool = false) -> Tensor {
  x.to_tensor(~requires_grad)
}

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

fn power(base : Float, exponent : Int) -> Float {
  if exponent == 0 {
    return 1.0
  }
  if exponent == 1 {
    return base
  }
  let mut result : Float = 1.0
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
  let output : FixedArray[Float] = FixedArray::make(length, 0.0)
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
        a.grad[i] = self.grad[i] * b.to_float() * power(a.value[i], b - 1)
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
    return tensors[0].reshape([1, ..tensors[0].shape])
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
  self.value[0].to_double()
}

pub fn Tensor::to_float(self : Tensor) -> Float {
  if self.shape.length() != 0 {
    abort("Tensor is not a scalar")
  }
  self.value[0]
}

pub fn Tensor::zeros[Shape : @shape.Shape](
  shape : Shape,
  ~requires_grad : Bool = false
) -> Tensor {
  let shape = shape.to_fixed_array()
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value : FixedArray[Float] = FixedArray::make(total, 0.0)
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
  let value : FixedArray[Float] = FixedArray::make(total, 1.0)
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

pub fn Tensor::value(self : Tensor) -> FixedArray[Float] {
  self.value
}
