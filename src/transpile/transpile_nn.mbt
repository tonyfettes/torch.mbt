///|
pub impl Transpile for @nn.Softmax with transpile(
  _self : @nn.Softmax,
  transpiler : Transpiler,
  variable : String
) -> Unit {
  transpiler.line("\{variable} = torch.nn.Softmax(dim=-1)")
}

///|
pub impl Transpile for @nn.Sigmoid with transpile(
  _self : @nn.Sigmoid,
  transpiler : Transpiler,
  variable : String
) -> Unit {
  transpiler.line("\{variable} = torch.nn.Sigmoid()")
}

///|
pub impl[Module : @nn.Module + Transpile] Transpile for @nn.Sequential[Module] with transpile(
  self : @nn.Sequential[Module],
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let module_variables = []
  for module_ in self.modules {
    let variable = transpiler.allocate()
    Transpile::transpile(module_, transpiler, variable)
    module_variables.push(variable)
  }
  transpiler.line("\{variable} = torch.nn.Sequential(")
  transpiler.open(fn() {
    for module_variable in module_variables {
      transpiler.line("\{module_variable},")
    }
  })
  transpiler.line(")")
}

///|
pub impl Transpile for @nn.ReLU with transpile(
  _self : @nn.ReLU,
  transpiler : Transpiler,
  variable : String
) -> Unit {
  transpiler.line("\{variable} = torch.nn.ReLU()")
}

///|
pub impl Transpile for @nn.Linear with transpile(
  self : @nn.Linear,
  transpiler : Transpiler,
  variable : String
) -> Unit {
  transpiler.line(
    "\{variable} = torch.nn.Linear(\{self.weight.shape[0]}, \{self.weight.shape[1]})",
  )
  transpiler.line("with torch.no_grad():")
  transpiler.open(fn() {
    transpiler.line(
      "\{variable}.weight = torch.nn.Parameter(torch.tensor(\{self.weight.to_json().stringify()}).transpose(-2, -1))",
    )
    transpiler.line(
      "\{variable}.bias = torch.nn.Parameter(torch.tensor(\{self.bias.to_json().stringify()}))",
    )
  })
}

///|
pub impl Transpile for @nn.Flatten with transpile(
  _self : @nn.Flatten,
  transpiler : Transpiler,
  variable : String
) -> Unit {
  transpiler.line("\{variable} = torch.nn.Flatten()")
}

///|
pub impl Transpile for @nn.Conv2d with transpile(
  self : @nn.Conv2d,
  transpiler : Transpiler,
  variable : String
) -> Unit {
  let kernel_size = self.weight[0][0].shape
  let kernel_size = (kernel_size[0], kernel_size[1])
  transpiler.line(
    "\{variable} = torch.nn.Conv2d(\{self.weight[1].length()}, \{self.weight.length()}, kernel_size=\{kernel_size}, padding=\{self.padding})",
  )
  transpiler.line("with torch.no_grad():")
  transpiler.open(fn() {
    transpiler.line(
      "\{variable}.weight = torch.nn.Parameter(torch.tensor(\{self.weight.to_json().stringify()}))",
    )
    transpiler.line(
      "\{variable}.bias = torch.nn.Parameter(torch.tensor(\{self.bias.to_json().stringify()}))",
    )
  })
}

///|
pub impl Transpile for @nn.AvgPool2d with transpile(
  self : @nn.AvgPool2d,
  transpiler : Transpiler,
  variable : String
) -> Unit {
  transpiler.line(
    "\{variable} = torch.nn.AvgPool2d(kernel_size=\{self.kernel_size}, stride=\{self.stride})",
  )
}
