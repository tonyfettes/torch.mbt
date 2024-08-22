pub trait Module {
  forward(Self, Tensor) -> Tensor
}

pub struct Sequential {
  modules : Array[Module]
}

pub fn Sequential::new(modules : Array[Module]) -> Sequential {
  Sequential::{ modules, }
}

pub fn Sequential::forward(
  self : Sequential,
  input : Tensor
) -> Tensor {
  loop input, self.modules[:] {
    input, [module, .. as modules] => continue module.forward(input), modules
    input, [] => input
  }
}

pub struct Linear {
  weights : Array[Tensor]
  bias : Tensor
}

let random_state : @random.RandomState = @random.init_state(seed=42)

pub fn Linear::new(in_features : Int, out_features : Int) -> Linear {
  Linear::{
    weights: Array::makei(
      out_features,
      fn {
        _ =>
          Array::makei(
            in_features,
            fn { _ => Value::var(@random.gen_double(random_state)) },
          )
      },
    ),
    bias: Array::makei(
      out_features,
      fn { _ => Value::var(@random.gen_double(random_state)) },
    ),
  }
}

pub fn Linear::forward(self : Linear, input : Tensor) -> Tensor {
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

fn reLU(input : Value) -> Value {
  if input.value > 0.0 {
    input
  } else {
    Value::val(0.0)
  }
}

pub fn ReLU::new() -> ReLU {
  ReLU::{  }
}

pub fn ReLU::forward(_self : ReLU, input : Tensor) -> Tensor {
  input.map(reLU)
}

pub struct ReLU6 {}

fn reLU6(input : Value) -> Value {
  if input.value > 6.0 {
    Value::val(6.0)
  } else if input.value > 0.0 {
    input
  } else {
    Value::val(0.0)
  }
}

pub fn ReLU6::new() -> ReLU6 {
  ReLU6::{  }
}

pub fn ReLU6::forward(_self : ReLU6, input : Tensor) -> Tensor {
  input.map(reLU6)
}

pub struct Softmax {}

pub fn Softmax::new() -> Softmax {
  Softmax::{  }
}

pub fn Softmax::forward(_self : Softmax, input : Tensor) -> Tensor {
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
  let target = [Value::val(1.0), Value::val(0.0)]
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i] * output[i].log()
  }
  inspect!(loss, content="7.000911466453775")
  loss.backward(loss.value * 0.01)
  inspect!(linear.weights[0][0], content="1.0699453327769946")
  inspect!(linear.weights[0][1], content="2.139890665553989")
  inspect!(linear.weights[1][0], content="2.930054667222996")
  inspect!(linear.weights[1][1], content="3.8601093344459927")
  inspect!(linear.bias[0], content="5.069945332776995")
  inspect!(linear.bias[1], content="5.930054667222996")
  let output = layers.forward(input)
  inspect!(output, content="[0.002106421695576897, 0.9978935783044232]")
}

test "Linear with ReLU" {
  let layers = Sequential::new(
    [Linear::new(2, 2), ReLU::new(), Linear::new(2, 2), Softmax::new()],
  )
  let input = [Value::val(1.0), Value::val(2.0)]
  let output = layers.forward(input)
  inspect!(output, content="[0.35707856359873796, 0.642921436401262]")
  let target = [Value::val(1.0), Value::val(0.0)]
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i] * output[i].log()
  }
  inspect!(loss, content="1.0297994553105834")
  loss.backward(loss.value * 0.01)
  let output = layers.forward(input)
  inspect!(output, content="[0.45122364611092397, 0.5487763538890761]")
}
