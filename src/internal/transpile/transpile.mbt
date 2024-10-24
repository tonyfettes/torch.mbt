pub struct Transpiler {
  mut ident : Int
  mut level : Int
  model : Array[(Int, String)]
  after : Array[(Int, String)]
  exports : Array[(String, String)]
}

pub fn Transpiler::new() -> Transpiler {
  Transpiler::{ ident: 0, level: 0, model: [], after: [], exports: [] }
}

pub fn Transpiler::allocate(self : Transpiler) -> String {
  let ident = self.ident
  self.ident = ident + 1
  "_layer\{ident}"
}

pub fn Transpiler::push_model(
  self : Transpiler,
  model : String
) -> Unit {
  let ident = self.level
  self.model.push((ident, model))
}

pub fn Transpiler::push_after(
  self : Transpiler,
  after : String
) -> Unit {
  let ident = self.level
  self.after.push((ident, after))
}

pub fn Transpiler::export(
  self : Transpiler,
  variable : String,
  identifier : String
) -> Unit {
  self.exports.push((variable, identifier))
}

pub fn Transpiler::to_string(self : Transpiler) -> String {
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

pub trait Transpile {
  transpile(Self, Transpiler) -> String
}

pub fn transpile[Module : Transpile](
  module : Module,
  export : String
) -> String {
  let transpiler = Transpiler::new()
  let variable = Transpile::transpile(module, transpiler)
  transpiler.export(variable, export)
  transpiler.to_string()
}
