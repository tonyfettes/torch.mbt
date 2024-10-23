pub impl[A : ToPyTorchSource, B : ToPyTorchSource] ToPyTorchSource for (A, B) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
) with transpile(self : (A, B, C), transpiler : PyTorchTranspiler) -> String {
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
) with transpile(self : (A, B, C, D), transpiler : PyTorchTranspiler) -> String {
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
) with transpile(self : (A, B, C, D, E), transpiler : PyTorchTranspiler) -> String {
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
) with transpile(self : (A, B, C, D, E, F), transpiler : PyTorchTranspiler) -> String {
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
) with transpile(self : (A, B, C, D, E, F, G), transpiler : PyTorchTranspiler) -> String {
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource, I : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource, I : ToPyTorchSource, J : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource, I : ToPyTorchSource, J : ToPyTorchSource, K : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource, I : ToPyTorchSource, J : ToPyTorchSource, K : ToPyTorchSource, L : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource, I : ToPyTorchSource, J : ToPyTorchSource, K : ToPyTorchSource, L : ToPyTorchSource, M : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource, I : ToPyTorchSource, J : ToPyTorchSource, K : ToPyTorchSource, L : ToPyTorchSource, M : ToPyTorchSource, N : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource, I : ToPyTorchSource, J : ToPyTorchSource, K : ToPyTorchSource, L : ToPyTorchSource, M : ToPyTorchSource, N : ToPyTorchSource, O : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
) with transpile(
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

pub impl[A : ToPyTorchSource, B : ToPyTorchSource, C : ToPyTorchSource, D : ToPyTorchSource, E : ToPyTorchSource, F : ToPyTorchSource, G : ToPyTorchSource, H : ToPyTorchSource, I : ToPyTorchSource, J : ToPyTorchSource, K : ToPyTorchSource, L : ToPyTorchSource, M : ToPyTorchSource, N : ToPyTorchSource, O : ToPyTorchSource, P : ToPyTorchSource] ToPyTorchSource for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
  P,
) with transpile(
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
