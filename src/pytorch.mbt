struct PyTorchTranspiler {
  mut ident : Int
  mut level : Int
  model : Array[(Int, String)]
  after : Array[(Int, String)]
}

fn PyTorchTranspiler::new() -> PyTorchTranspiler {
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

pub trait ToPyTorchSource {
  transpile(Self, PyTorchTranspiler) -> String
  to_string(Self) -> String
}

impl ToPyTorchSource with to_string(self : Self) -> String {
  let transpiler = PyTorchTranspiler::new()
  ToPyTorchSource::transpile(self, transpiler) |> ignore
  let model = []
  for line in transpiler.model {
    let (level, line) = line
    model.push(String::make(level * 4, ' ') + line)
  }
  let model = String::concat(model, separator="\n")
  let after = []
  for line in transpiler.after {
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
