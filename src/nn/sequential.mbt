pub(all) struct Sequential[Module] {
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
    input, [module_, .. as modules] => continue module_.forward(input), modules
    input, [] => input
  }
}

pub fn Sequential::parameters(
  self : Sequential[Module]
) -> Iter[@tensor.Tensor] {
  self.modules.iter().flat_map(fn(module_) { module_.parameters() })
}

pub fn Sequential::output(self : Sequential[Module], logger : Logger) -> Unit {
  for module_ in self.modules {
    Show::output(module_, logger)
    logger.write_string(",\n")
  }
}

pub fn Sequential::to_string(self : Sequential[Module]) -> String {
  let logger = StringBuilder::new()
  self.output(logger)
  logger.to_string()
}
