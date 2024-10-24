pub impl Transpile for @nn.Softmax with transpile(
  _self : @nn.Softmax,
  transpiler : Transpiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.Softmax(dim=-1)")
  variable
}

pub impl Transpile for @nn.Sigmoid with transpile(
  _self : @nn.Sigmoid,
  transpiler : Transpiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.Sigmoid()")
  variable
}

pub impl[Module : @nn.Module + Transpile] Transpile for @nn.Sequential[Module] with transpile(
  self : @nn.Sequential[Module],
  transpiler : Transpiler
) -> String {
  let module_variables = []
  for module in self.modules {
    module_variables.push(Transpile::transpile(module, transpiler))
  }
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

pub impl Transpile for @nn.ReLU with transpile(
  _self : @nn.ReLU,
  transpiler : Transpiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.ReLU()")
  variable
}

pub impl Transpile for @nn.Linear with transpile(
  self : @nn.Linear,
  transpiler : Transpiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model(
    "\{variable} = torch.nn.Linear(\{self.weight.shape[0]}, \{self.weight.shape[1]})",
  )
  transpiler.push_after(
    "\{variable}.weight = torch.nn.Parameter(torch.tensor(\{self.weight.to_json().stringify()}).transpose(-2, -1))",
  )
  transpiler.push_after(
    "\{variable}.bias = torch.nn.Parameter(torch.tensor(\{self.bias.to_json().stringify()}))",
  )
  variable
}

pub impl Transpile for @nn.Flatten with transpile(
  _self : @nn.Flatten,
  transpiler : Transpiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.Flatten()")
  variable
}

pub impl Transpile for @nn.Conv2d with transpile(
  self : @nn.Conv2d,
  transpiler : Transpiler
) -> String {
  let variable = transpiler.allocate()
  let kernel_size = self.weight[0][0].shape
  let kernel_size = (kernel_size[0], kernel_size[1])
  transpiler.push_model(
    "\{variable} = torch.nn.Conv2d(\{self.weight[1].length()}, \{self.weight.length()}, kernel_size=\{kernel_size}, padding=\{self.padding})",
  )
  transpiler.push_after(
    "\{variable}.weight = torch.nn.Parameter(torch.tensor(\{self.weight.to_json().stringify()}))",
  )
  transpiler.push_after(
    "\{variable}.bias = torch.nn.Parameter(torch.tensor(\{self.bias.to_json().stringify()}))",
  )
  variable
}

pub impl Transpile for @nn.AvgPool2d with transpile(
  self : @nn.AvgPool2d,
  transpiler : Transpiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model(
    "\{variable} = torch.nn.AvgPool2d(kernel_size=\{self.kernel_size}, stride=\{self.stride})",
  )
  variable
}
