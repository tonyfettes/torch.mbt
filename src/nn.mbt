pub trait Module {
  forward(Self, Array[Value]) -> Array[Value]
}

pub struct Sequential {
  modules : Array[Module]
}

pub fn Sequential::new(modules : Array[Module]) -> Sequential {
  Sequential::{ modules, }
}

pub fn Sequential::forward(
  self : Sequential,
  input : Array[Value]
) -> Array[Value] {
  loop input, self.modules[:] {
    input, [module, .. as modules] => continue module.forward(input), modules
    input, [] => input
  }
}

pub struct Linear {
  weights : Array[Array[Value]]
  bias : Array[Value]
}

pub fn Linear::new(in_features : Int, out_features : Int) -> Linear {
  Linear::{
    weights: Array::makei(
      out_features,
      fn { _ => Array::makei(in_features, fn { _ => Value::var(0.0) }) },
    ),
    bias: Array::makei(out_features, fn { _ => Value::var(0.0) }),
  }
}

pub fn Linear::forward(self : Linear, input : Array[Value]) -> Array[Value] {
  let output = []
  for row = 0; row < self.weights.length(); row = row + 1 {
    output.push(self.bias[row])
    for column = 0; column < self.weights[row].length(); column = column + 1 {
      output[row] = output[row] + self.weights[row][column] * input[column]
    }
  }
  output
}

pub struct ReLU {}

pub fn reLU(input : Value) -> Value {
  if input.value > 0.0 {
    input
  } else {
    Value::val(0.0)
  }
}

pub fn ReLU::new() -> ReLU {
  ReLU::{  }
}

pub fn ReLU::forward(_self : ReLU, input : Array[Value]) -> Array[Value] {
  input.map(reLU)
}

pub struct Softmax {}

pub fn Softmax::new() -> Softmax {
  Softmax::{  }
}

pub fn Softmax::forward(_self : Softmax, input : Array[Value]) -> Array[Value] {
  let mut sum = Value::val(0.0)
  for value in input {
    sum = sum + value.exp()
  }
  input.map(fn { value => value / sum })
}

test "" {
  let layers = Sequential::new([
    Linear::new(2, 3),
    ReLU::new(),
    Linear::new(3, 2),
    Softmax::new(),
  ])
  let input = [Value::val(1.0), Value::val(2.0)]
  let output = layers.forward(input)
  inspect!(output, content="[0.0, 0.0]")
}
