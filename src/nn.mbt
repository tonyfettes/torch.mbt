pub trait Module: Show {
  forward(Self, Tensor) -> Tensor
  parameters(Self) -> Iter[Tensor]
}

struct Sequential {
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

pub fn Sequential::parameters(self : Sequential) -> Iter[Tensor] {
  self.modules.iter().flat_map(fn { module => module.parameters() })
}

pub fn Sequential::output(self : Sequential, logger : Logger) -> Unit {
  for module in self.modules {
    Show::output(module, logger)
    logger.write_string(",\n")
  }
}

pub fn Sequential::to_string(self : Sequential) -> String {
  let logger = Buffer::new()
  self.output(logger)
  logger.to_string()
}

pub struct ReLU {} derive(Show, ToJson, FromJson)

pub fn ReLU::new() -> ReLU {
  ReLU::{  }
}

pub fn ReLU::output(_self : ReLU, logger : Logger) -> Unit {
  logger.write_string("ReLU()")
}

pub fn ReLU::forward(_self : ReLU, input : Tensor) -> Tensor {
  input.reLU()
}

pub fn ReLU::parameters(_self : ReLU) -> Iter[Tensor] {
  Iter::empty()
}

pub struct ReLU6 {}

pub struct Softmax {} derive(Show, ToJson, FromJson)

pub fn Softmax::new() -> Softmax {
  Softmax::{  }
}

pub fn Softmax::output(_self : Softmax, logger : Logger) -> Unit {
  logger.write_string("Softmax()")
}

pub fn Softmax::forward(_self : Softmax, input : Tensor) -> Tensor {
  let exp = input.exp()
  let sum = exp.sum()
  exp / sum
}

pub fn Softmax::parameters(_self : Softmax) -> Iter[Tensor] {
  Iter::empty()
}

test "Softmax" {
  let input = tensor([1.0, 2.0, 3.0])
  let softmax = Softmax::new()
  let output = softmax.forward(input)
  inspect!(
    output,
    content="[0.09003057317038046, 0.24472847105479767, 0.6652409557748219]",
  )
  let target = tensor([1.0, 0.0, 0.0])
  let mut loss = tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss + target[i] * output[i]
  }
  inspect!(loss, content="0.09003057317038046")
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
  let input = tensor([1.0, 1.0, 1.0, 1.0])
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.20922499162142535, 0.25805928787959304, 0.2551680112861502, 0.27754770921283145]",
  )
  let target = tensor([1.0, 0.0, 0.0, 0.0])
  let mut loss = tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i].to_tensor() * output[i].log()
  }
  inspect!(loss, content="1.5643450911089625")
  let optimizer = SGD::new(layers.parameters(), learning_rate=0.1)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.24395465491177973, 0.24583749392308907, 0.24326207530914587, 0.2669457758559854]",
  )
}

pub trait Loss {
  forward(Self, Tensor, Tensor) -> TensorValue
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
) -> Tensor {
  if input.length() != target.length() {
    abort("Input and target tensors should have the same length")
  }
  let diff = input - target
  diff.pow(2).sum() / tensor(diff.length().to_double())
}

pub struct CrossEntropyLoss {}

pub fn CrossEntropyLoss::new() -> CrossEntropyLoss {
  CrossEntropyLoss::{  }
}

pub fn CrossEntropyLoss::output(
  _self : CrossEntropyLoss,
  logger : Logger
) -> Unit {
  logger.write_string("CrossEntropyLoss()")
}

pub fn CrossEntropyLoss::forward(
  _self : CrossEntropyLoss,
  input : Tensor,
  target : Tensor
) -> Tensor {
  let diff = -target * input.log()
  diff.sum() / tensor(input.length().to_double())
}

test "Linear Regression" {
  let model = Linear::new(1, 1)
  with_no_grad(
    fn() {
      model.weight[0][0] = 0.0
      model.bias[0] = 0.0
    },
  )
  let optimizer = SGD::new(model.parameters(), learning_rate=0.1)
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
    let model_output = model.forward(tensor([input]))
    let target_output = target([input])
    let mse = MSELoss::new()
    let loss = mse.forward(model_output, tensor(target_output))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  }
  inspect!(model.weight[0][0], content="1.981059015531822")
  inspect!(model.bias[0], content="3.01049253711016")
}

struct Sigmoid {} derive(Show, ToJson, FromJson)

pub fn Sigmoid::new() -> Sigmoid {
  Sigmoid::{  }
}

pub fn Sigmoid::forward(_self : Sigmoid, input : Tensor) -> Tensor {
  input.sigmoid()
}

pub fn Sigmoid::parameters(_self : Sigmoid) -> Iter[Tensor] {
  return Iter::empty()
}

test "Sigmoid" {
  seed(b"12345678223456783234567842345678")
  let input = Uniform::new(-1, 1).sample([3, 3])
  inspect!(
    input,
    content="[[-0.29557028390369, -0.8177422144361883, -0.8997844018869048], [-0.1545487762608706, -0.5520054503667071, -0.09482517741472507], [0.6609056774831252, 0.21990249887250046, 0.4313355035432884]]",
  )
  inspect!(
    input.sigmoid(),
    content="[[0.4266407195423779, 0.30624313516780693, 0.28909480486844663], [0.4614395278462282, 0.36539925445953875, 0.47631145323784313], [0.6594638069803208, 0.5547551522746634, 0.6061925292136984]]",
  )
}

struct Flatten {
  start : Int
  end : Int?
} derive(Show, ToJson, FromJson)

pub fn Flatten::new(~start : Int = 1, ~end? : Int) -> Flatten {
  Flatten::{ start, end }
}

pub fn Flatten::forward(self : Flatten, input : Tensor) -> Tensor {
  input.flatten(start=self.start, end?=self.end)
}

pub fn Flatten::parameters(_self : Flatten) -> Iter[Tensor] {
  Iter::empty()
}
