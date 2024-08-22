pub trait Module: Show {
  forward(Self, Tensor) -> Tensor
}

pub struct Sequential {
  modules : Array[Module]
}

pub fn Sequential::new(modules : Array[Module]) -> Sequential {
  Sequential::{ modules, }
}

pub fn Sequential::forward(self : Sequential, input : Tensor) -> Tensor {
  loop input, self.modules[:] {
    input, [module, .. as modules] => continue module.forward(input), modules
    input, [] => input
  }
}

pub fn Sequential::output(self : Sequential, logger : Logger) -> Unit {
  for module in self.modules {
    module.output(logger)
    logger.write_string(",\n")
  }
}

pub fn Sequential::to_string(self : Sequential) -> String {
  let logger = Buffer::new()
  self.output(logger)
  logger.to_string()
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
            fn { _ => Value::var(@random.gen_double(random_state) / 2.0 - 0.5) },
          )
      },
    ),
    bias: Array::makei(
      out_features,
      fn { _ => Value::var(@random.gen_double(random_state) / 2.0 - 0.5) },
    ),
  }
}

pub fn Linear::forward(self : Linear, input : Tensor) -> Tensor {
  let output = []
  for row = 0; row < self.weights.length(); row = row + 1 {
    let products = []
    for column = 0; column < self.weights[row].length(); column = column + 1 {
      products.push(self.weights[row][column] * input[column])
    }
    output.push(self.bias[row] + Value::sum(products))
  }
  output
}

pub fn Linear::output(self : Linear, logger : Logger) -> Unit {
  logger.write_string("Linear(\n")
  for row = 0; row < self.weights.length(); row = row + 1 {
    logger.write_string("  ")
    Show::output(self.weights[row], logger)
    logger.write_string("\t| ")
    Show::output(self.bias[row], logger)
    logger.write_char('\n')
  }
  logger.write_string(")")
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

pub fn ReLU::output(_self : ReLU, logger : Logger) -> Unit {
  logger.write_string("ReLU()")
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

pub fn ReLU6::output(_self : ReLU6, logger : Logger) -> Unit {
  logger.write_string("ReLU6()")
}

pub fn ReLU6::forward(_self : ReLU6, input : Tensor) -> Tensor {
  input.map(reLU6)
}

pub struct Softmax {}

pub fn Softmax::new() -> Softmax {
  Softmax::{  }
}

pub fn Softmax::output(_self : Softmax, logger : Logger) -> Unit {
  logger.write_string("Softmax()")
}

pub fn Softmax::forward(_self : Softmax, input : Tensor) -> Tensor {
  let exp = input.map(fn { value => value.exp() })
  let sum = Value::sum(exp)
  let output = []
  for value in exp {
    output.push(value / sum)
  }
  output
}

test "Softmax" {
  let input = [Value::var(1.0), Value::var(2.0), Value::var(3.0)]
  let softmax = Softmax::new()
  let output = softmax.forward(input)
  inspect!(
    output,
    content="[0.09003057317038046, 0.24472847105479767, 0.6652409557748219]",
  )
  let target = Tensor::val([1.0, 0.0, 0.0])
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss + target[i] * output[i]
  }
  inspect!(loss, content="0.09003057317038046")
  loss.backward(1.0)
  let output = softmax.forward(input)
  inspect!(
    output,
    content="[0.07980224135677778, 0.24068997746912693, 0.6795077811740953]",
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
  let layers = Sequential::new([linear])
  let input = [Value::val(1.0), Value::val(2.0)]
  let output = layers.forward(input)
  inspect!(output, content="[10.0, 17.0]")
  let target = [Value::val(16.0), Value::val(16.0)]
  for i = 0; i < output.length(); i = i + 1 {
    let diff = output[i] - target[i]
    let loss = diff
    loss.backward(0.1 * loss.value)
  }
  inspect!(linear.weights[0][0], content="1.6")
  inspect!(linear.weights[0][1], content="3.2")
  inspect!(linear.weights[1][0], content="2.9")
  inspect!(linear.weights[1][1], content="3.8")
  inspect!(linear.bias[0], content="5.6")
  inspect!(linear.bias[1], content="5.9")
  let output = layers.forward(input)
  inspect!(output, content="[13.6, 16.4]")
}

test "Linear with ReLU" {
  let linear = Linear::new(2, 2)
  linear.weights[0][0] = Value::var(1.0)
  linear.weights[0][1] = Value::var(2.0)
  linear.weights[1][0] = Value::var(3.0)
  linear.weights[1][1] = Value::var(4.0)
  linear.bias[0] = Value::var(5.0)
  linear.bias[1] = Value::var(6.0)
  let layers = Sequential::new([linear, ReLU::new()])
  let input = [Value::val(2.0), Value::val(2.0)]
  let output = layers.forward(input)
  inspect!(output, content="[11.0, 20.0]")
  let target = [Value::val(1.0), Value::val(0.0)]
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i] * output[i].log()
  }
  inspect!(loss, content="-2.3978952727983707")
  loss.backward(1.0)
  let output = layers.forward(input)
  inspect!(output, content="[11.818181818181817, 20.0]")
}

test "Linear with Softmax" {
  let linear = Linear::new(2, 2)
  linear.weights[0][0] = Value::var(2.0)
  linear.weights[0][1] = Value::var(2.0)
  linear.weights[1][0] = Value::var(2.0)
  linear.weights[1][1] = Value::var(2.0)
  linear.bias[0] = Value::var(1.0)
  linear.bias[1] = Value::var(2.0)
  let layers = Sequential::new([linear, Softmax::new()])
  let input = [Value::val(2.0), Value::val(2.0)]
  let output = layers.forward(input)
  let target = Tensor::val([1.0, 0.0])
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i] * output[i].log()
  }
  inspect!(loss, content="1.3132616875182228")
  loss.backward(1.0)
  let output = layers.forward(input)
  inspect!(output, content="[0.9999947593218812, 5.24067811873532e-6]")
}

test "MPL" {
  let layers = Sequential::new(
    [
      Linear::new(4, 8),
      ReLU::new(),
      Linear::new(8, 8),
      ReLU::new(),
      Linear::new(8, 4),
      Softmax::new(),
    ],
  )
  let input = [
    Value::val(1.0),
    Value::val(1.0),
    Value::val(1.0),
    Value::val(1.0),
  ]
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.3980194395832311, 0.17007562599442944, 0.2750160036912784, 0.15688893073106108]",
  )
  let target = Tensor::val([1.0, 0.0, 0.0, 0.0])
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i] * output[i].log()
  }
  inspect!(loss, content="0.9212544317165409")
  loss.backward(loss.value * 0.01)
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.40280662885309937, 0.16918197833897097, 0.27229714290358314, 0.15571424990434649]",
  )
}
