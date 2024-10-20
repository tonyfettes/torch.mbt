pub struct PyTorchTranspiler {
  mut ident : Int
  mut level : Int
  model : Array[(Int, String)]
  after : Array[(Int, String)]
  exports : Array[(String, String)]
}

pub fn PyTorchTranspiler::new() -> PyTorchTranspiler {
  PyTorchTranspiler::{ ident: 0, level: 0, model: [], after: [], exports: [] }
}

pub fn PyTorchTranspiler::allocate(self : PyTorchTranspiler) -> String {
  let ident = self.ident
  self.ident = ident + 1
  "_layer\{ident}"
}

pub fn PyTorchTranspiler::push_model(
  self : PyTorchTranspiler,
  model : String
) -> Unit {
  let ident = self.level
  self.model.push((ident, model))
}

pub fn PyTorchTranspiler::push_after(
  self : PyTorchTranspiler,
  after : String
) -> Unit {
  let ident = self.level
  self.after.push((ident, after))
}

pub fn PyTorchTranspiler::export(
  self : PyTorchTranspiler,
  variable : String,
  identifier : String
) -> Unit {
  self.exports.push((variable, identifier))
}

pub fn PyTorchTranspiler::to_string(self : PyTorchTranspiler) -> String {
  let model = []
  for line in self.model {
    let (level, line) = line
    model.push(String::make(level * 4, ' ') + line)
  }
  let model = String::concat(model, separator="\n")
  let after = []
  for line in self.after {
    let (level, line) = line
    after.push(String::make(level * 4, ' ') + line)
  }
  let after = String::concat(after, separator="\n    ")
  let exports = []
  let __all__ = []
  for export in self.exports {
    let (variable, identifier) = export
    exports.push("\{identifier} = \{variable}")
    __all__.push("\"\{identifier}\"")
  }
  let exports = String::concat(exports, separator="\n")
  let __all__ = String::concat(__all__, separator=", ")
  $|import torch
  $|
  $|\{model}
  $|
  $|with torch.no_grad():
  $|    \{after}
  $|
  $|\{exports}
  $|
  $|__all__ = [\{__all__}]
}

pub trait ToPyTorchSource {
  transpile(Self, PyTorchTranspiler) -> String
}

pub fn transpile[Module : ToPyTorchSource](
  module : Module,
  export : String
) -> String {
  let transpiler = PyTorchTranspiler::new()
  let variable = ToPyTorchSource::transpile(module, transpiler)
  transpiler.export(variable, export)
  transpiler.to_string()
}
