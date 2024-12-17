///|
pub(all) struct Sequential[Module] {
  modules : Array[Module]
}

///|
pub fn Sequential::new[Module : Module](
  modules : Array[Module]
) -> Sequential[Module] {
  Sequential::{ modules, }
}

///|
pub fn Sequential::forward[Module : Module](
  self : Sequential[Module],
  input : @tensor.Tensor[Float]
) -> @tensor.Tensor[Float] {
  loop input, self.modules[:] {
    input, [module_, .. as modules] => continue module_.forward(input), modules
    input, [] => input
  }
}

///|
pub fn Sequential::parameters[Module : Module](
  self : Sequential[Module]
) -> Iter[@tensor.Tensor[Float]] {
  self.modules.iter().flat_map(fn(module_) { module_.parameters() })
}

///|
pub fn Sequential::output[Module : Module](
  self : Sequential[Module],
  logger : &Logger
) -> Unit {
  for module_ in self.modules {
    Show::output(module_, logger)
    logger.write_string(",\n")
  }
}

///|
pub fn Sequential::to_string[Module : Module](
  self : Sequential[Module]
) -> String {
  let logger = StringBuilder::new()
  self.output(logger)
  logger.to_string()
}
