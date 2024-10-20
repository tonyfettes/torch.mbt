pub impl[A : Module, B : Module] Module for (A, B) with forward(
  self : (A, B),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
}

pub impl[A : Module, B : Module] Module for (A, B) with parameters(
  self : (A, B)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
}

pub impl[A : Module, B : Module] ToPyTorchSource for (A, B) with transpile(
  self : (A, B),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
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

pub impl[A : Module, B : Module, C : Module] Module for (A, B, C) with forward(
  self : (A, B, C),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
}

pub impl[A : Module, B : Module, C : Module] Module for (A, B, C) with parameters(
  self : (A, B, C)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
}

pub impl[A : Module, B : Module, C : Module] ToPyTorchSource for (A, B, C) with transpile(
  self : (A, B, C),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module] Module for (A, B, C, D) with forward(
  self : (A, B, C, D),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
}

pub impl[A : Module, B : Module, C : Module, D : Module] Module for (A, B, C, D) with parameters(
  self : (A, B, C, D)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module] ToPyTorchSource for (A, B, C, D) with transpile(
  self : (A, B, C, D),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module] Module for (A, B, C, D, E) with forward(
  self : (A, B, C, D, E),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module] Module for (A, B, C, D, E) with parameters(
  self : (A, B, C, D, E)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module] ToPyTorchSource for (A, B, C, D, E) with transpile(
  self : (A, B, C, D, E),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module] Module for (A, B, C, D, E, F) with forward(
  self : (A, B, C, D, E, F),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module] Module for (A, B, C, D, E, F) with parameters(
  self : (A, B, C, D, E, F)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module] ToPyTorchSource for (A, B, C, D, E, F) with transpile(
  self : (A, B, C, D, E, F),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module] Module for (A, B, C, D, E, F, G) with forward(
  self : (A, B, C, D, E, F, G),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module] Module for (A, B, C, D, E, F, G) with parameters(
  self : (A, B, C, D, E, F, G)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module] ToPyTorchSource for (A, B, C, D, E, F, G) with transpile(
  self : (A, B, C, D, E, F, G),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module] Module for (A, B, C, D, E, F, G, H) with forward(
  self : (A, B, C, D, E, F, G, H),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module] Module for (A, B, C, D, E, F, G, H) with parameters(
  self : (A, B, C, D, E, F, G, H)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H) with transpile(
  self : (A, B, C, D, E, F, G, H),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module] Module for (A, B, C, D, E, F, G, H, I) with forward(
  self : (A, B, C, D, E, F, G, H, I),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module] Module for (A, B, C, D, E, F, G, H, I) with parameters(
  self : (A, B, C, D, E, F, G, H, I)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H, I) with transpile(
  self : (A, B, C, D, E, F, G, H, I),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.8, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module] Module for (A, B, C, D, E, F, G, H, I, J) with forward(
  self : (A, B, C, D, E, F, G, H, I, J),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module] Module for (A, B, C, D, E, F, G, H, I, J) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H, I, J) with transpile(
  self : (A, B, C, D, E, F, G, H, I, J),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.8, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.9, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module] Module for (A, B, C, D, E, F, G, H, I, J, K) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module] Module for (A, B, C, D, E, F, G, H, I, J, K) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H, I, J, K) with transpile(
  self : (A, B, C, D, E, F, G, H, I, J, K),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.8, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.9, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.10, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H, I, J, K, L) with transpile(
  self : (A, B, C, D, E, F, G, H, I, J, K, L),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.8, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.9, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.10, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.11, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
  |> forward_(self.12)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
  |> Iter::concat(self.12.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H, I, J, K, L, M) with transpile(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.8, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.9, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.10, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.11, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.12, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
  |> forward_(self.12)
  |> forward_(self.13)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
  |> Iter::concat(self.12.parameters())
  |> Iter::concat(self.13.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H, I, J, K, L, M, N) with transpile(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.8, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.9, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.10, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.11, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.12, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.13, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
  |> forward_(self.12)
  |> forward_(self.13)
  |> forward_(self.14)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
  |> Iter::concat(self.12.parameters())
  |> Iter::concat(self.13.parameters())
  |> Iter::concat(self.14.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O) with transpile(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.8, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.9, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.10, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.11, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.12, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.13, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.14, transpiler))
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

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module, P : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) with forward(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input
  |> forward_(self.0)
  |> forward_(self.1)
  |> forward_(self.2)
  |> forward_(self.3)
  |> forward_(self.4)
  |> forward_(self.5)
  |> forward_(self.6)
  |> forward_(self.7)
  |> forward_(self.8)
  |> forward_(self.9)
  |> forward_(self.10)
  |> forward_(self.11)
  |> forward_(self.12)
  |> forward_(self.13)
  |> forward_(self.14)
  |> forward_(self.15)
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module, P : Module] Module for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) with parameters(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)
) -> Iter[@tensor.Tensor] {
  self.0.parameters()
  |> Iter::concat(self.1.parameters())
  |> Iter::concat(self.2.parameters())
  |> Iter::concat(self.3.parameters())
  |> Iter::concat(self.4.parameters())
  |> Iter::concat(self.5.parameters())
  |> Iter::concat(self.6.parameters())
  |> Iter::concat(self.7.parameters())
  |> Iter::concat(self.8.parameters())
  |> Iter::concat(self.9.parameters())
  |> Iter::concat(self.10.parameters())
  |> Iter::concat(self.11.parameters())
  |> Iter::concat(self.12.parameters())
  |> Iter::concat(self.13.parameters())
  |> Iter::concat(self.14.parameters())
  |> Iter::concat(self.15.parameters())
}

pub impl[A : Module, B : Module, C : Module, D : Module, E : Module, F : Module, G : Module, H : Module, I : Module, J : Module, K : Module, L : Module, M : Module, N : Module, O : Module, P : Module] ToPyTorchSource for (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) with transpile(
  self : (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P),
  transpiler : PyTorchTranspiler
) -> String {
  let module_variables = []
  module_variables.push(ToPyTorchSource::transpile(self.0, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.1, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.2, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.3, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.4, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.5, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.6, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.7, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.8, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.9, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.10, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.11, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.12, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.13, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.14, transpiler))
  module_variables.push(ToPyTorchSource::transpile(self.15, transpiler))
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