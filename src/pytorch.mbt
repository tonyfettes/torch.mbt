struct PyTorchTranspiler {
  mut ident : Int
  mut level : Int
  model : Array[(Int, String)]
  after : Array[(Int, String)]
}

pub fn PyTorchTranspiler::new() -> PyTorchTranspiler {
  PyTorchTranspiler::{ ident: 0, level: 0, model: [], after: [] }
}

fn PyTorchTranspiler::allocate(self : PyTorchTranspiler) -> String {
  let ident = self.ident
  self.ident = ident + 1
  "layer\{ident}"
}

fn PyTorchTranspiler::push_model(
  self : PyTorchTranspiler,
  model : String
) -> Unit {
  let ident = self.level
  self.model.push((ident, model))
}

fn PyTorchTranspiler::push_after(
  self : PyTorchTranspiler,
  after : String
) -> Unit {
  let ident = self.level
  self.after.push((ident, after))
}

pub fn PyTorchTranspiler::transpile[Module: Module](self : PyTorchTranspiler, module : Module) -> String {
  module.transpile(self)
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
  $|import torch
  $|
  $|\{model}
  $|
  $|with torch.no_grad():
  $|    \{after}
}

pub trait ToPyTorchSource {
  transpile(Self, PyTorchTranspiler) -> String
}
