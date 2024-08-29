pub enum Graph {
  Val
  Var
  Add(Value, Value)
  Mul(Value, Value)
  Div(Value, Value)
  Neg(Value)
  Exp(Value)
  Log(Value)
  Sum(Tensor)
  Pow(Value, Int)
}

pub struct Value {
  mut value : Double
  priv graph : Graph
  priv mut ref : Int
  mut grad : Double
}

pub fn Value::output(self : Value, logger : Logger) -> Unit {
  Show::output(self.value, logger)
}

pub fn Value::to_string(self : Value) -> String {
  Show::to_string(self.value)
}

pub fn Value::to_debug(self : Value) -> String {
  match self.graph {
    Val => {
      let debug = self.to_string()
      "(\{debug})"
    }
    Var => {
      let debug = self.to_string()
      "(\{debug})"
    }
    Add(a, b) => {
      let a_debug = a.to_debug()
      let b_debug = b.to_debug()
      "(\{a_debug} + \{b_debug})"
    }
    Mul(a, b) => {
      let a_debug = a.to_debug()
      let b_debug = b.to_debug()
      "(\{a_debug} * \{b_debug})"
    }
    Div(a, b) => {
      let a_debug = a.to_debug()
      let b_debug = b.to_debug()
      "(\{a_debug} / \{b_debug})"
    }
    Neg(a) => {
      let a_debug = a.to_debug()
      "(-\{a_debug})"
    }
    Exp(a) => {
      let a_debug = a.to_debug()
      "exp \{a_debug}"
    }
    Log(a) => {
      let a_debug = a.to_debug()
      "log \{a_debug}"
    }
    Sum(xs) => {
      let mut debug = ""
      for i, x in xs.0 {
        let x_debug = x.to_debug()
        if i != xs.length() - 1 {
          debug = debug + x_debug + " + "
        } else {
          debug = debug + x_debug
        }
      }
      "(\{debug})"
    }
    Pow(a, b) => {
      let a_debug = a.to_debug()
      "(\{a_debug} ^ \{b})"
    }
  }
}

pub fn Value::val(value : Double) -> Value {
  Value::{ value, graph: Graph::Val, ref: 0, grad: 0.0 }
}

pub fn Value::var(value : Double) -> Value {
  Value::{ value, graph: Graph::Var, ref: 0, grad: 0.0 }
}

pub fn Value::sum(values : Tensor) -> Value {
  let mut sum = 0.0
  for value in values {
    sum = sum + value.value
  }
  Value::{ value: sum, graph: Graph::Sum(values), ref: 0, grad: 0.0 }
}

fn Value::ref(self : Value) -> Unit {
  self.ref += 1
  if self.ref > 1 {
    return
  }
  self.grad = 0.0
  match self.graph {
    Val => ()
    Var => ()
    Add(a, b) => {
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
    Sum(xs) =>
      for x in xs {
        x.ref()
      }
    Pow(x, _) => x.ref()
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
  inspect!(power(2.0, 2), content="4")
  inspect!(power(2.0, 3), content="8")
  inspect!(power(2.0, 4), content="16")
  inspect!(power(2.0, 5), content="32")
  inspect!(power(3.0, 6), content="729")
}

fn Value::propagate(self : Value, value : Double) -> Unit {
  self.grad += value
  self.ref -= 1
  if self.ref != 0 {
    return
  }
  match self.graph {
    Val => return
    Var => self.value -= self.grad
    Add(a, b) => {
      a.propagate(self.grad)
      b.propagate(self.grad)
    }
    Mul(a, b) => {
      let a_value = a.value
      let b_value = b.value
      a.propagate(self.grad * b_value)
      b.propagate(self.grad * a_value)
    }
    Div(a, b) => {
      let a_value = a.value
      let b_value = b.value
      a.propagate(self.grad / b_value)
      b.propagate(-self.grad * a_value / b_value / b_value)
    }
    Neg(a) => a.propagate(-self.grad)
    Exp(a) => a.propagate(self.grad * self.value)
    Log(a) => {
      let a_value = a.value
      a.propagate(self.grad / a_value)
    }
    Sum(xs) =>
      for x in xs {
        x.propagate(self.grad)
      }
    Pow(a, b) => {
      let a_value = a.value
      a.propagate(self.grad * b.to_double() * power(a_value, b - 1))
    }
  }
}

fn Value::check(self : Value) -> Unit {
  if self.ref != 0 {
    let debug = self.to_debug()
    abort("ref != 0 found: \{debug}")
  }
  match self.graph {
    Val => ()
    Var => ()
    Add(a, b) => {
      a.check()
      b.check()
    }
    Mul(a, b) => {
      a.check()
      b.check()
    }
    Div(a, b) => {
      a.check()
      b.check()
    }
    Neg(a) => a.check()
    Exp(a) => a.check()
    Log(a) => a.check()
    Sum(xs) =>
      for x in xs {
        x.check()
      }
    Pow(a, _) => a.check()
  }
}

pub fn Value::backward(self : Value, value : Double) -> Unit {
  self.ref()
  self.propagate(value)
  self.check()
}

pub fn Value::op_neg(self : Value) -> Value {
  Value::{ value: -self.value, graph: Graph::Neg(self), ref: 0, grad: 0.0 }
}

pub fn Value::op_add(self : Value, other : Value) -> Value {
  Value::{
    value: self.value + other.value,
    graph: Graph::Add(self, other),
    ref: 0,
    grad: 0.0,
  }
}

pub fn Value::op_sub(self : Value, other : Value) -> Value {
  self + -other
}

pub fn Value::op_mul(self : Value, other : Value) -> Value {
  Value::{
    value: self.value * other.value,
    graph: Graph::Mul(self, other),
    ref: 0,
    grad: 0.0,
  }
}

pub fn Value::op_div(self : Value, other : Value) -> Value {
  Value::{
    value: self.value / other.value,
    graph: Graph::Div(self, other),
    ref: 0,
    grad: 0.0,
  }
}

pub fn Value::exp(self : Value) -> Value {
  Value::{ value: self.value.exp(), graph: Graph::Exp(self), ref: 0, grad: 0.0 }
}

pub fn Value::log(self : Value) -> Value {
  Value::{ value: self.value.ln(), graph: Graph::Log(self), ref: 0, grad: 0.0 }
}

pub fn Value::pow(self : Value, exponent : Int) -> Value {
  Value::{
    value: power(self.value, exponent),
    graph: Graph::Pow(self, exponent),
    ref: 0,
    grad: 0.0,
  }
}

test "Value::backward - op_neg" {
  let v1 = Value::var(5.0)
  let neg = -v1
  inspect!(neg, content="-5")
  neg.ref()
  inspect!(v1.ref, content="1")
  neg.propagate(1.0)
  inspect!(v1.ref, content="0")
  inspect!(v1, content="6")
}

test "Value::backward - op_add" {
  let v1 = Value::var(5.0)
  let v2 = Value::var(4.0)
  let sum = v1 + v2
  inspect!(sum, content="9")
  sum.ref()
  inspect!(v1.ref, content="1")
  inspect!(v2.ref, content="1")
  sum.propagate(1.0)
  inspect!(v1.ref, content="0")
  inspect!(v2.ref, content="0")
  inspect!(v1, content="4")
  inspect!(v2, content="3")
}

test "Value::backward - op_mul" {
  let v1 = Value::var(5.0)
  let v2 = Value::var(4.0)
  let product = v1 * v2
  inspect!(product, content="20")
  product.ref()
  inspect!(v1.ref, content="1")
  inspect!(v2.ref, content="1")
  product.propagate(1.0)
  inspect!(v1.ref, content="0")
  inspect!(v2.ref, content="0")
  inspect!(v1, content="1")
  inspect!(v2, content="-1")
}

test "Value::backward - op_div" {
  let v1 = Value::var(5.0)
  let v2 = Value::var(4.0)
  let quotient = v1 / v2
  inspect!(quotient, content="1.25")
  quotient.ref()
  inspect!(v1.ref, content="1")
  inspect!(v2.ref, content="1")
  quotient.propagate(1.0)
  inspect!(v1.ref, content="0")
  inspect!(v2.ref, content="0")
  inspect!(v1, content="4.75")
  inspect!(v2, content="4.3125")
}

test "Value::backward - exp" {
  let v1 = Value::var(5.0)
  let exp = v1.exp()
  inspect!(exp, content="148.4131591025766")
  exp.ref()
  inspect!(v1.ref, content="1")
  exp.propagate(1.0)
  inspect!(v1.ref, content="0")
  inspect!(v1, content="-143.4131591025766")
}

test "Value::backward - log" {
  let v1 = Value::var(5.0)
  let log = v1.log()
  inspect!(log, content="1.6094379124341003")
  log.ref()
  inspect!(v1.ref, content="1")
  log.propagate(1.0)
  inspect!(v1.ref, content="0")
  inspect!(v1, content="4.8")
}

test "Value::backward - sum" {
  let v = [Value::var(1.0), Value::var(2.0), Value::var(3.0)]
  let sum = Value::sum(v)
  inspect!(sum, content="6")
  sum.ref()
  inspect!(v[0].ref, content="1")
  inspect!(v[1].ref, content="1")
  inspect!(v[2].ref, content="1")
  sum.propagate(1.0)
  inspect!(v[0].ref, content="0")
  inspect!(v[1].ref, content="0")
  inspect!(v[2].ref, content="0")
  inspect!(v[0], content="0")
  inspect!(v[1], content="1")
  inspect!(v[2], content="2")
}
