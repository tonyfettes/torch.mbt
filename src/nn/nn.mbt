pub trait Module: Show {
  forward(Self, @tensor.Tensor) -> @tensor.Tensor
  parameters(Self) -> Iter[@tensor.Tensor]
}

struct Sequential[Module] {
  modules : Array[Module]
}

pub fn Sequential::new[Module: Module](modules : Array[Module]) -> Sequential[Module] {
  Sequential::{ modules, }
}

pub fn Sequential::forward(
  self : Sequential[Module],
  input : @tensor.Tensor
) -> @tensor.Tensor {
  loop input, self.modules[:] {
    input, [module, .. as modules] => continue module.forward(input), modules
    input, [] => input
  }
}

pub fn Sequential::parameters(
  self : Sequential[Module]
) -> Iter[@tensor.Tensor] {
  self.modules.iter().flat_map(fn { module => module.parameters() })
}

pub fn Sequential::output(self : Sequential[Module], logger : Logger) -> Unit {
  for module in self.modules {
    Show::output(module, logger)
    logger.write_string(",\n")
  }
}

pub fn Sequential::to_string(self : Sequential[Module]) -> String {
  let logger = StringBuilder::new()
  self.output(logger)
  logger.to_string()
}

pub impl[Module : Module + @transpile.ToPyTorchSource] @transpile.ToPyTorchSource for Sequential[
  Module,
] with transpile(
  self : Sequential[Module],
  transpiler : @transpile.PyTorchTranspiler
) -> String {
  let module_variables = []
  for module in self.modules {
    module_variables.push(
      @transpile.ToPyTorchSource::transpile(module, transpiler),
    )
  }
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.Sequential(")
  transpiler.level = transpiler.level + 1
  for module_variable in module_variables {
    transpiler.push_model("\{module_variable},")
  }
  transpiler.level = transpiler.level - 1
  transpiler.push_model(")")
  variable
}

pub struct ReLU {} derive(Show, ToJson, FromJson)

pub fn ReLU::new() -> ReLU {
  ReLU::{  }
}

pub fn ReLU::output(_self : ReLU, logger : Logger) -> Unit {
  logger.write_string("ReLU()")
}

pub fn ReLU::forward(_self : ReLU, input : @tensor.Tensor) -> @tensor.Tensor {
  input.reLU()
}

pub fn ReLU::parameters(_self : ReLU) -> Iter[@tensor.Tensor] {
  Iter::empty()
}

pub impl @transpile.ToPyTorchSource for ReLU with transpile(
  _self : ReLU,
  transpiler : @transpile.PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.ReLU()")
  variable
}

pub struct ReLU6 {}

pub struct Softmax {} derive(Show, ToJson, FromJson)

pub fn Softmax::new() -> Softmax {
  Softmax::{  }
}

pub fn Softmax::output(_self : Softmax, logger : Logger) -> Unit {
  logger.write_string("Softmax()")
}

pub fn Softmax::forward(
  _self : Softmax,
  input : @tensor.Tensor
) -> @tensor.Tensor {
  let exp = input.exp()
  let sum = exp.sum()
  exp / sum
}

pub fn Softmax::parameters(_self : Softmax) -> Iter[@tensor.Tensor] {
  Iter::empty()
}

pub impl @transpile.ToPyTorchSource for Softmax with transpile(
  _self : Softmax,
  transpiler : @transpile.PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.Softmax(dim=-1)")
  variable
}

test "Softmax" {
  let input = @tensor.tensor([1.0, 2.0, 3.0])
  let softmax = Softmax::new()
  let output = softmax.forward(input)
  inspect!(
    output,
    content="[0.09003057317038046, 0.24472847105479767, 0.6652409557748219]",
  )
  let target = @tensor.tensor([1.0, 0.0, 0.0])
  let mut loss = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss + target[i] * output[i]
  }
  inspect!(loss, content="0.09003057317038046")
}

test "MLP" {
  @distributions.seed(b"12345678223456783234567842345678")
  let layers : Sequential[Module] = Sequential::new(
    [
      Linear::new(4, 8),
      ReLU::new(),
      Linear::new(8, 8),
      ReLU::new(),
      Linear::new(8, 4),
      Softmax::new(),
    ],
  )
  let input = @tensor.tensor([1.0, 1.0, 1.0, 1.0])
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.20922499162142535, 0.25805928787959304, 0.2551680112861502, 0.27754770921283145]",
  )
  let target = @tensor.tensor([1.0, 0.0, 0.0, 0.0])
  let mut loss = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i].to_tensor() * output[i].log()
  }
  inspect!(loss, content="1.5643450911089625")
  let optimizer = @optim.SGD::new(layers.parameters(), learning_rate=0.1)
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
  forward(Self, @tensor.Tensor, @tensor.Tensor) -> @tensor.TensorValue
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
  input : @tensor.Tensor,
  target : @tensor.Tensor
) -> @tensor.Tensor {
  if input.length() != target.length() {
    abort("Input and target tensors should have the same length")
  }
  let diff = input - target
  diff.pow(2).mean()
}

test "MSELoss" {
  let input = @tensor.tensor([[1.0, 2.0, 3.0]])
  let target = @tensor.tensor([[4.0, 5.0, 6.0]])
  let criterion = MSELoss::new()
  inspect!(criterion.forward(input, input), content="0")
  inspect!(criterion.forward(input, target), content="9")
}

struct Flatten {
  start : Int
  end : Int?
} derive(Show, ToJson, FromJson)

pub fn Flatten::new(~start : Int = 1, ~end? : Int) -> Flatten {
  Flatten::{ start, end }
}

pub fn Flatten::forward(
  self : Flatten,
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input.flatten(start=self.start, end?=self.end)
}

pub fn Flatten::parameters(_self : Flatten) -> Iter[@tensor.Tensor] {
  Iter::empty()
}

pub impl @transpile.ToPyTorchSource for Flatten with transpile(
  _self : Flatten,
  transpiler : @transpile.PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.Flatten()")
  variable
}

fn forward_[Module : Module](
  tensor : @tensor.Tensor,
  module : Module
) -> @tensor.Tensor {
  module.forward(tensor)
}

// pub fn transpile[Module : Module](
//   module : Module,
//   export : String
// ) -> String {
//   let transpiler = @transpile.PyTorchTranspiler::new()
//   let variable = @transpile.ToPyTorchSource::transpile(module, transpiler)
//   transpiler.export(variable, export)
//   transpiler.to_string()
// }
