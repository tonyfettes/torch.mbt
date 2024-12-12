///|
pub impl[A : Transpile, B : Transpile] Transpile for (A, B) with transpile(
  self : (A, B),
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile] Transpile for (A, B, C) with transpile(
  self : (A, B, C),
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile] Transpile for (
  A,
  B,
  C,
  D,
) with transpile(
  self : (A, B, C, D),
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile] Transpile for (
  A,
  B,
  C,
  D,
  E,
) with transpile(
  self : (A, B, C, D, E),
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile] Transpile for (
  A,
  B,
  C,
  D,
  E,
  F,
) with transpile(
  self : (A, B, C, D, E, F),
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile] Transpile for (
  A,
  B,
  C,
  D,
  E,
  F,
  G,
) with transpile(
  self : (A, B, C, D, E, F, G),
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile, I : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.8, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile, I : Transpile, J : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.8, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.9, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile, I : Transpile, J : Transpile, K : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.8, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.9, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.10, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile, I : Transpile, J : Transpile, K : Transpile, L : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.8, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.9, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.10, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.11, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile, I : Transpile, J : Transpile, K : Transpile, L : Transpile, M : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.8, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.9, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.10, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.11, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.12, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile, I : Transpile, J : Transpile, K : Transpile, L : Transpile, M : Transpile, N : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.8, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.9, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.10, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.11, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.12, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.13, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile, I : Transpile, J : Transpile, K : Transpile, L : Transpile, M : Transpile, N : Transpile, O : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.8, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.9, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.10, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.11, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.12, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.13, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.14, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}

///|
pub impl[A : Transpile, B : Transpile, C : Transpile, D : Transpile, E : Transpile, F : Transpile, G : Transpile, H : Transpile, I : Transpile, J : Transpile, K : Transpile, L : Transpile, M : Transpile, N : Transpile, O : Transpile, P : Transpile] Transpile for (
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
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.0, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.1, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.2, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.3, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.4, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.5, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.6, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.7, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.8, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.9, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.10, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.11, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.12, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.13, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.14, transpiler, module_variable)
  module_variables.push(module_variable)
  let module_variable = transpiler.allocate()
  Transpile::transpile(self.15, transpiler, module_variable)
  module_variables.push(module_variable)
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(
    fn() {
      for module_variable in module_variables {
        transpiler.line("\{module_variable},")
      }
    },
  )
  transpiler.line(")")
}
