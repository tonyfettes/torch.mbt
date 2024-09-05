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
  weight : Tensor
  bias : Tensor
}

pub fn Linear::of(weight : Tensor, bias : Tensor) -> Linear {
  Linear::{ weight, bias }
}

pub fn Linear::new(
  in_features : Int,
  out_features : Int,
  ~bias : Bool = true,
  ~weight_distribution? : Continuous,
  ~bias_distribution? : Continuous
) -> Linear {
  let in_features_sqrt = in_features.to_double().sqrt()
  let weight_distribution = match weight_distribution {
    None => {
      let in_features_sqrt_inv = 1.0 / in_features_sqrt
      Uniform::new(-in_features_sqrt_inv, in_features_sqrt_inv) as Continuous
    }
    Some(distrib) => distrib
  }
  let bias_distribution = match bias_distribution {
    None => {
      let bound = 1.0 / in_features_sqrt
      Uniform::new(-bound, bound) as Continuous
    }
    Some(distrib) => distrib
  }
  Linear::{
    weight: weight_distribution.sample(~shape=[in_features, out_features]),
    bias: bias_distribution.sample(~shape=[out_features]),
  }
}

pub fn Linear::forward(self : Linear, input : Tensor) -> Tensor {
  input.matmul(self.weight) + self.bias
}

pub fn Linear::output(self : Linear, logger : Logger) -> Unit {
  logger.write_string("Linear(\n")
  for row = 0; row < self.weight.length(); row = row + 1 {
    logger.write_string("  ")
    Show::output(self.weight[row], logger)
    logger.write_string("\t| ")
    Show::output(self.bias[row], logger)
    logger.write_char('\n')
  }
  logger.write_string(")")
}

test "Linear" {
  let linear = Linear::new(2, 2)
  linear.weight[0][0] = Value::var(1.0)
  linear.weight[0][1] = Value::var(2.0)
  linear.weight[1][0] = Value::var(3.0)
  linear.weight[1][1] = Value::var(4.0)
  linear.bias[0] = Value::var(5.0)
  linear.bias[1] = Value::var(6.0)
  let layers = Sequential::new([linear])
  let input = [Value::val(1.0), Value::val(2.0)]
  let output = layers.forward(input)
  inspect!(output, content="[10, 17]")
  let target = [Value::val(16.0), Value::val(16.0)]
  for i = 0; i < output.length(); i = i + 1 {
    let diff = output[i] - target[i]
    let loss = diff
    loss.backward(0.1 * loss.value)
  }
  inspect!(linear.weight[0][0], content="1.6")
  inspect!(linear.weight[0][1], content="3.2")
  inspect!(linear.weight[1][0], content="2.9")
  inspect!(linear.weight[1][1], content="3.8")
  inspect!(linear.bias[0], content="5.6")
  inspect!(linear.bias[1], content="5.9")
  let output = layers.forward(input)
  inspect!(output, content="[13.6, 16.4]")
}

test "Linear with ReLU" {
  let linear = Linear::new(2, 2)
  linear.weight[0][0] = Value::var(1.0)
  linear.weight[0][1] = Value::var(2.0)
  linear.weight[1][0] = Value::var(3.0)
  linear.weight[1][1] = Value::var(4.0)
  linear.bias[0] = Value::var(5.0)
  linear.bias[1] = Value::var(6.0)
  let layers = Sequential::new([linear, ReLU::new()])
  let input = [Value::val(2.0), Value::val(2.0)]
  let output = layers.forward(input)
  inspect!(output, content="[11, 20]")
  let target = [Value::val(1.0), Value::val(0.0)]
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i] * output[i].log()
  }
  inspect!(loss, content="-2.3978952727983707")
  loss.backward(1.0)
  let output = layers.forward(input)
  inspect!(output, content="[11.818181818181817, 20]")
}

test "Linear with Softmax" {
  let linear = Linear::new(2, 2)
  linear.weight[0][0] = Value::var(2.0)
  linear.weight[0][1] = Value::var(2.0)
  linear.weight[1][0] = Value::var(2.0)
  linear.weight[1][1] = Value::var(2.0)
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
  inspect!(output, content="[0.9999947593218812, 0.00000524067811873532]")
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

test "MLP" {
  seed(b"12345678223456783234567842345678")
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
    content="[0.18384932308483545, 0.27209037641072775, 0.27304055950350237, 0.27101974100093446]",
  )
  let target = Tensor::val([1.0, 0.0, 0.0, 0.0])
  let mut loss = Value::val(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i] * output[i].log()
  }
  inspect!(loss, content="1.6936387531299375")
  loss.backward(loss.value * 0.01)
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.19019143854877638, 0.2703981560947781, 0.2696450961817953, 0.2697653091746504]",
  )
}

pub trait Loss {
  forward(Self, Tensor, Tensor) -> Value
}

pub struct MSELoss {}

pub fn MSELoss::new() -> MSELoss {
  MSELoss::{  }
}

pub fn MSELoss::output(_self : MSELoss, logger : Logger) -> Unit {
  logger.write_string("MSELoss()")
}

pub fn MSELoss::forward(
  _self : MSELoss,
  input : Tensor,
  target : Tensor
) -> Value {
  let diff = []
  for i = 0; i < input.length(); i = i + 1 {
    diff.push(input[i] - target[i])
  }
  let loss = Value::sum(diff.map(fn { value => value.pow(2) }))
  loss / Value::val(diff.length().to_double())
}

pub struct CrossEntropyLoss {}

pub fn CrossEntropyLoss::new() -> CrossEntropyLoss {
  CrossEntropyLoss::{  }
}

pub fn CrossEntropyLoss::output(_self : CrossEntropyLoss, logger : Logger) -> Unit {
  logger.write_string("CrossEntropyLoss()")
}

pub fn CrossEntropyLoss::forward(
  _self : CrossEntropyLoss,
  input : Tensor,
  target : Tensor
) -> Value {
  let diff = []
  for i = 0; i < input.length(); i = i + 1 {
    diff.push(- target[i] * input[i].log())
  }
  Value::sum(diff)
}

test "Linear Regression" {
  let model = Linear::new(1, 1)
  model.weight[0][0] = Value::var(0.0)
  model.bias[0] = Value::var(0.0)
  let target = fn(x : Array[Double]) { [2.0 * x[0] + 3] }
  let inputs = [
    0.8208251, 0.90000966, 0.28106993, 0.39323831, 0.43506248, 0.4969614, 0.52247973,
    0.49798604, 0.43939068, 0.3734946, 0.4441855, 0.17318048, 0.53219596, 0.63448068,
    0.27115805, 0.04605697, 0.62318401, 0.33910485, 0.69114928, 0.85872608, 0.70759624,
    0.40058802, 0.1777967, 0.92748176, 0.94344838, 0.78039178, 0.33418037, 0.43679556,
    0.39154924, 0.10587376, 0.05049539, 0.29797588, 0.00861857, 0.40571597, 0.21740392,
    0.65157348, 0.08771972, 0.72849501, 0.08581929, 0.8128746, 0.18798559, 0.30142073,
    0.32481787, 0.52926784, 0.1449449, 0.20553479, 0.38707897, 0.40028778, 0.11176918,
    0.76972532, 0.42597432, 0.01741988, 0.80524058, 0.35739401, 0.24440017, 0.41526613,
    0.99033731, 0.81271161, 0.46265435, 0.95218477, 0.62453272, 0.15694713, 0.20849319,
    0.97591847, 0.48845406, 0.72073021, 0.52854909, 0.41928837, 0.40103961, 0.66694493,
    0.79072028, 0.19673913, 0.45183849, 0.02185975, 0.15932374, 0.89000227, 0.06060862,
    0.45457933, 0.86395899, 0.54407434, 0.36498587, 0.33118886, 0.97118758, 0.85580425,
    0.53489337, 0.71204262, 0.72440047, 0.63693202, 0.57291731, 0.04470077, 0.74890034,
    0.17283568, 0.01892377, 0.88189203, 0.5364349, 0.56212932, 0.68814933, 0.02615991,
    0.98899348, 0.25727347,
  ]
  for input in inputs {
    let model_output = model.forward(Tensor::val([input]))
    let target_output = target([input])
    let mse = MSELoss::new()
    let loss = mse.forward(model_output, Tensor::val(target_output))
    loss.backward(0.1)
  }
  inspect!(model.weight[0][0], content="1.981059015531822")
  inspect!(model.bias[0], content="3.01049253711016")
}
