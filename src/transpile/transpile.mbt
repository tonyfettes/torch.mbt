struct Transpiler {
  mut ident : Int
  mut level : Int
  lines : Array[(Int, String)]
  exports : Array[String]
}

pub fn Transpiler::new() -> Transpiler {
  Transpiler::{ ident: 0, level: 0, lines: [], exports: [] }
}

pub fn Transpiler::allocate(self : Transpiler) -> String {
  let ident = self.ident
  self.ident = ident + 1
  "_layer\{ident}"
}

pub fn Transpiler::line(self : Transpiler, line : String) -> Unit {
  self.lines.push((self.level, line))
}

pub fn Transpiler::open(self : Transpiler, f : () -> Unit) -> Unit {
  self.level = self.level + 1
  f()
  self.level = self.level - 1
}

pub fn Transpiler::to_string(self : Transpiler) -> String {
  let lines = []
  for line in self.lines {
    let (level, line) = line
    lines.push(String::make(level * 4, ' ') + line)
  }
  let content = String::concat(lines, separator="\n")
  let __all__ = []
  for export in self.exports {
    let variable = export
    __all__.push("\"\{variable}\"")
  }
  let __all__ = String::concat(__all__, separator=", ")
  $|import torch
  $|
  $|\{content}
  $|
  $|__all__ = [\{__all__}]
}

pub trait Transpile {
  transpile(Self, Transpiler, String) -> Unit
}

pub fn transpile[Module : Transpile](
  module : Module,
  export : String
) -> String {
  let transpiler = Transpiler::new()
  Transpile::transpile(module, transpiler, export)
  transpiler.exports.push(export)
  transpiler.to_string()
}
