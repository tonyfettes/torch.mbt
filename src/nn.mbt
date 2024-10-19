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
