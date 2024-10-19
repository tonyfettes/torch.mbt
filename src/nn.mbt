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
  let logger = StringBuilder::new()
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
  diff.pow(2).sum() / tensor(diff.value.length().to_double())
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
  let max = input.max()
  let sub_max = input - max
  let log_soft_max = sub_max - sub_max.exp().sum().log()
  let nl = -target * log_soft_max
  nl.sum()
}

test "CrossEntropyLoss" {
  let input = tensor([0.0, 1.0e3, 0.0])
  let target = tensor([0.3, 0.3, 0.4])
  let loss = CrossEntropyLoss::new()
  let output = loss.forward(input, target)
  inspect!(output, content="700")
}

test "CrossEntropyLoss/Random" {
  let input = tensor(
    [
      -0.14630347333285626, -0.11041421512364283, -0.12559252186003866, -0.13513121430614242,
      -0.1788202792999719, -0.1412897765465771, -0.12746655654191671, -0.12665480060556913,
      -0.14233145825454024, -0.1360269781704941,
    ],
  )
  let target = {
    let target = FixedArray::make(10, 0.0)
    target[6] = 1.0
    tensor(target)
  }
  let criterion = CrossEntropyLoss::new()
  let output = criterion.forward(input, target)
  inspect!(output, content="2.293193741636991")
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
