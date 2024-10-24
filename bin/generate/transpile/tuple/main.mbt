fn build_pytorch_transpile(length : Int) -> String {
  let type_parameters = @generate/nn/tuple.build_type_parameters(
    length,
    constraint="Transpile",
  )
  let tuple_type = @generate/nn/tuple.build_tuple_type(length)
  let push_children = []
  for i in 0..<length {
    push_children.push(
      "module_variables.push(Transpile::transpile(self.\{i}, transpiler))",
    )
  }
  let push_children = String::concat(push_children, separator="\n  ")
  $|pub impl[\{type_parameters}] Transpile for \{tuple_type} with transpile(
  $|  self : \{tuple_type},
  #|  transpiler : Transpiler
  #|) -> String {
  #|  let module_variables = []
  $|  \{push_children}
  #|  let variable = transpiler.allocate()
  #|  transpiler.push_model("\{variable} = torch.nn.Sequential(")
  #|  transpiler.level = transpiler.level + 1
  #|  for module_variable in module_variables {
  #|    transpiler.push_model("\{module_variable},")
  #|  }
  #|  transpiler.level = transpiler.level - 1
  #|  transpiler.push_model(")")
  #|  variable
  #|}
}

fn main {
  let functions = []
  for i in 2..=16 {
    functions.push(build_pytorch_transpile(i))
  }
  let content = String::concat(functions, separator="\n\n")
  try {
    @generate.write_on_change!(
      path="../src/transpile/tuple.mbt",
      ~content,
    )
  } catch {
    error =>
      $|Failed to write content to src/transpile/tuple.mbt: \{Show::to_string(error)}
      |> println
  }
}
