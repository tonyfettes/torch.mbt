pub enum Graph {
  Val
  Var
  Add(Value, Value)
  Mul(Value, Value)
  Div(Value, Value)
  Neg(Value)
  Exp(Value)
  Log(Value)
}

pub struct Value {
  mut value: Double
  graph: Graph
}

pub fn Value::val(value: Double) -> Value {
  Value::{
    value: value,
    graph: Graph::Val
  }
}

pub fn Value::var(value: Double) -> Value {
  Value::{
    value: value,
    graph: Graph::Var
  }
}

pub fn Value::backward(self : Value, value : Double) -> Unit {
  match self.graph {
    Val => ()
    Var => self.value -= value
    Add(a, b) => {
      a.backward(value)
      b.backward(value)
      self.value -= value
    }
    Mul(a, b) => {
      a.backward(value * b.value)
      b.backward(value * a.value)
      self.value -= value
    }
    Div(a, b) => {
      a.backward(value / b.value)
      b.backward(-value * a.value / b.value / b.value)
      self.value -= value
    }
    Neg(a) => {
      a.backward(-value)
      self.value -= value
    }
    Exp(a) => {
      a.backward(value * self.value)
      self.value -= value
    }
    Log(a) => {
      a.backward(value / self.value)
      self.value -= value
    }
  }
}

test "Value::backward - op_neg" {
  let v1 = Value::var(5.0)
  let neg = -v1
  neg.backward(1.0)
  inspect!(v1.value, content="6.0")
}

test "Value::backward - op_add" {
  let v1 = Value::var(5.0)
  let v2 = Value::var(4.0)
  let sum = v1 + v2
  sum.backward(1.0)
  inspect!(v1.value, content="4.0")
  inspect!(v2.value, content="3.0")
}

test "Value::backward - op_mul" {
  let v1 = Value::var(5.0)
  let v2 = Value::var(4.0)
  let product = v1 * v2
  product.backward(1.0)
  inspect!(v1.value, content="1.0")
  inspect!(v2.value, content="3.0")
}


pub fn Value::op_neg(self : Value) -> Value {
  Value::{
    value: -self.value,
    graph: Graph::Neg(self)
  }
}

pub fn Value::op_add(self : Value, other : Value) -> Value {
  Value::{
    value: self.value + other.value,
    graph: Graph::Add(self, other)
  }
}

pub fn Value::op_mul(self : Value, other : Value) -> Value {
  Value::{
    value: self.value * other.value,
    graph: Graph::Mul(self, other)
  }
}

pub fn Value::op_div(self : Value, other : Value) -> Value {
  Value::{
    value: self.value / other.value,
    graph: Graph::Div(self, other)
  }
}

pub fn Value::exp(self : Value) -> Value {
  Value::{
    value: self.value.exp(),
    graph: Graph::Exp(self)
  }
}

pub fn Value::log(self : Value) -> Value {
  Value::{
    value: self.value.ln(),
    graph: Graph::Log(self)
  }
}
