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
  let input_exp = input.map(fn { value => value.exp() })
  let mut sum = Value::val(0.0)
  for value in input_exp {
    sum = sum + value
  }
  input_exp.map(fn { value => value / sum })
}

test "Softmax" {
  let input = [Value::val(1.0), Value::val(2.0), Value::val(3.0)]
  let softmax = Softmax::new()
  let output = softmax.forward(input)
  inspect!(
    output,
    content="[0.09003057317038046, 0.24472847105479767, 0.6652409557748219]",
  )
}

test "Linear" {
  let linear = Linear::new(2, 2)
  linear.weights[0][0] = Value::var(1.0)
  linear.weights[0][1] = Value::var(2.0)
  linear.weights[1][0] = Value::var(3.0)
  linear.weights[1][1] = Value::var(4.0)
  linear.bias[0] = Value::var(5.0)
  linear.bias[1] = Value::var(6.0)
  let layers = Sequential::new([linear, Softmax::new()])
  let input = [Value::val(1.0), Value::val(2.0)]
  let output = layers.forward(input)
  inspect!(output, content="[9.110511944006454e-4, 0.9990889488055994]")
  let target = [Value::val(0.9), Value::val(0.1)]
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss + (output[i].log() - target[i].log()) * (output[i].log() - target[i].log())
  }
  inspect!(loss, content="52.84632439813396")
  loss.backward(loss.value * 0.001)
  inspect!(linear.weights[0][0], content="1.3643103694563203")
  inspect!(linear.weights[0][1], content="2.7286207389126407")
  inspect!(linear.weights[1][0], content="2.635689630543608")
  inspect!(linear.weights[1][1], content="3.271379261087216")
  inspect!(linear.bias[0], content="5.3643103694563194")
  inspect!(linear.bias[1], content="5.6356896305436095")
  let output = layers.forward(input)
  inspect!(output, content="[0.06734067442631975, 0.9326593255736804]")
}
