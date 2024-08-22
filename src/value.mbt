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
}

pub struct Value {
  mut value : Double
  graph : Graph
  mut ref : Int
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
    Val => self.to_string()
    Var => self.to_string()
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
      "exp\{a_debug}"
    }
    Log(a) => {
      let a_debug = a.to_debug()
      "log\{a_debug}"
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
      "sum(\{debug})"
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
    value.ref += 1
    sum = sum + value.value
  }
  Value::{ value: sum, graph: Graph::Sum(values), ref: 0, grad: 0.0 }
}

fn Value::deref(self : Value) -> Unit {
  if (self.ref == 0) {
    let debug = self.to_debug()
    abort("Value::deref: Reference count is zero, \{debug}")
  }
  self.ref -= 1
}

pub fn Value::backward(self : Value, value : Double) -> Unit {
  self.grad += value
  if (self.ref != 0) {
    return
  }
  match self.graph {
    Val => return
    Var => ()
    Add(a, b) => {
      a.deref()
      a.backward(self.grad)
      b.deref()
      b.backward(self.grad)
    }
    Mul(a, b) => {
      let a_value = a.value
      let b_value = b.value
      a.deref()
      a.backward(self.grad * b_value)
      b.deref()
      b.backward(self.grad * a_value)
    }
    Div(a, b) => {
      let a_value = a.value
      let b_value = b.value
      a.deref()
      a.backward(self.grad / b_value)
      b.deref()
      b.backward(-self.grad * a_value / b_value / b_value)
    }
    Neg(a) => {
      a.deref()
      a.backward(-self.grad)
    }
    Exp(a) => {
      a.deref()
      a.backward(self.grad * self.value)
    }
    Log(a) => {
      let a_value = a.value
      a.deref()
      a.backward(self.grad / a_value)
    }
    Sum(xs) => {
      for x in xs {
        x.deref()
        x.backward(self.grad)
      }
    }
  }
  self.value -= self.grad
}

pub fn Value::op_neg(self : Value) -> Value {
  self.ref += 1
  Value::{ value: -self.value, graph: Graph::Neg(self), ref: 0, grad: 0.0 }
}

pub fn Value::op_add(self : Value, other : Value) -> Value {
  self.ref += 1
  other.ref += 1
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
  self.ref += 1
  other.ref += 1
  Value::{
    value: self.value * other.value,
    graph: Graph::Mul(self, other),
    ref: 0,
    grad: 0.0,
  }
}

pub fn Value::op_div(self : Value, other : Value) -> Value {
  self.ref += 1
  other.ref += 1
  Value::{
    value: self.value / other.value,
    graph: Graph::Div(self, other),
    ref: 0,
    grad: 0.0,
  }
}

pub fn Value::exp(self : Value) -> Value {
  self.ref += 1
  Value::{ value: self.value.exp(), graph: Graph::Exp(self), ref: 0, grad: 0.0 }
}

pub fn Value::log(self : Value) -> Value {
  self.ref += 1
  Value::{ value: self.value.ln(), graph: Graph::Log(self), ref: 0, grad: 0.0 }
}

test "Value::backward - op_neg" {
  let v1 = Value::var(5.0)
  let neg = -v1
  inspect!(neg, content="-5.0")
  inspect!(v1.ref, content="1")
  neg.backward(1.0)
  inspect!(v1, content="6.0")
}

test "Value::backward - op_add" {
  let v1 = Value::var(5.0)
  let v2 = Value::var(4.0)
  let sum = v1 + v2
  inspect!(sum, content="9.0")
  sum.backward(1.0)
  inspect!(v1, content="4.0")
  inspect!(v2, content="3.0")
}

test "Value::backward - op_mul" {
  let v1 = Value::var(5.0)
  let v2 = Value::var(4.0)
  let product = v1 * v2
  inspect!(product, content="20.0")
  product.backward(1.0)
  inspect!(v1, content="1.0")
  inspect!(v2, content="-1.0")
}

test "Value::backward - op_div" {
  let v1 = Value::var(5.0)
  let v2 = Value::var(4.0)
  let quotient = v1 / v2
  inspect!(quotient, content="1.25")
  quotient.backward(1.0)
  inspect!(v1, content="4.75")
  inspect!(v2, content="4.3125")
}

test "Value::backward - exp" {
  let v1 = Value::var(5.0)
  let exp = v1.exp()
  inspect!(exp, content="148.4131591025766")
  exp.backward(1.0)
  inspect!(v1, content="-143.4131591025766")
}

test "Value::backward - log" {
  let v1 = Value::var(5.0)
  let log = v1.log()
  inspect!(log, content="1.6094379124341003")
  log.backward(1.0)
  inspect!(v1, content="4.8")
}
