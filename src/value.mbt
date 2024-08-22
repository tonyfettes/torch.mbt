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
  mut value : Double
  graph : Graph
}

pub fn Value::output(self : Value, logger : Logger) -> Unit {
  Show::output(self.value, logger)
}

pub fn Value::val(value : Double) -> Value {
  Value::{ value, graph: Graph::Val }
}

pub fn Value::var(value : Double) -> Value {
  Value::{ value, graph: Graph::Var }
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
      let a_value = a.value
      let b_value = b.value
      a.backward(value * b_value)
      b.backward(value * a_value)
      self.value -= value
    }
    Div(a, b) => {
      let a_value = a.value
      let b_value = b.value
      a.backward(value / b_value)
      b.backward(-value * a_value / b_value / b_value)
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

pub fn Value::op_neg(self : Value) -> Value {
  Value::{ value: -self.value, graph: Graph::Neg(self) }
}

pub fn Value::op_add(self : Value, other : Value) -> Value {
  Value::{ value: self.value + other.value, graph: Graph::Add(self, other) }
}

pub fn Value::op_mul(self : Value, other : Value) -> Value {
  Value::{ value: self.value * other.value, graph: Graph::Mul(self, other) }
}

pub fn Value::op_div(self : Value, other : Value) -> Value {
  Value::{ value: self.value / other.value, graph: Graph::Div(self, other) }
}

pub fn Value::exp(self : Value) -> Value {
  Value::{ value: self.value.exp(), graph: Graph::Exp(self) }
}

pub fn Value::log(self : Value) -> Value {
  Value::{ value: self.value.ln(), graph: Graph::Log(self) }
}

test "Value::backward - op_neg" {
  let v1 = Value::var(5.0)
  let neg = -v1
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
  inspect!(v1, content="4.378665065440388")
}
