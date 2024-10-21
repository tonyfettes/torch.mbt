struct Sequential[Module] {
  modules : Array[Module]
}

pub fn Sequential::new[Module : Module](
  modules : Array[Module]
) -> Sequential[Module] {
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
