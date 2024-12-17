///|
fn build_pytorch_transpile(length : Int) -> String {
  let type_parameters = @generate/nn/tuple.build_type_parameters(
    length,
    constraint="Transpile",
  )
  let tuple_type = @generate/nn/tuple.build_tuple_type(length)
  let push_children = []
  for i in 0..<length {
    push_children.push("let module_variable = transpiler.allocate()")
    push_children.push(
      "Transpile::transpile(self.\{i}, transpiler, module_variable)",
    )
    push_children.push("module_variables.push(module_variable)")
  }
  let push_children = String::concat(push_children, separator="\n  ")
  $|pub impl[\{type_parameters}] Transpile for \{tuple_type} with transpile(
  $|  self : \{tuple_type},
  #|  transpiler : Transpiler,
  #|  variable : String
  #|) -> Unit {
  #|  let module_variables = []
  $|  \{push_children}
  #|  transpiler.line("\{variable} = torch.nn.Sequential(")
  #|  transpiler.open(
  #|    fn() {
  #|      for module_variable in module_variables {
  #|        transpiler.line("\{module_variable},")
  #|      }
  #|    },
  #|  )
  #|  transpiler.line(")")
  #|}
}

///|
fn main {
  let functions = []
  for i in 2..=16 {
    functions.push(build_pytorch_transpile(i))
  }
  let content = String::concat(functions, separator="\n\n")
  try {
    @generate.write_on_change!(path="../src/transpile/tuple.mbt", content~)
  } catch {
    error =>
      $|Failed to write content to src/transpile/tuple.mbt: \{Show::to_string(error)}
      |> println
  }
}
