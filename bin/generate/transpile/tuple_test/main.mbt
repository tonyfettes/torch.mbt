///|
fn build_pytorch_transpile(length : Int) -> String {
  let push_children = []
  for i in 0..<length {
    push_children.push("@nn.Linear::new(1, 1),")
  }
  let push_children = String::concat(push_children, separator="\n    ")
  #|  let model = (
  $|    \{push_children}
  #|  )
  $|  Transpile::transpile(model, transpiler, "tuple\{length}")
}

///|
fn main {
  let functions = []
  for i in 2..=16 {
    functions.push(build_pytorch_transpile(i))
  }
  let content = String::concat(functions, separator="\n")
  let content =
    #|test (it : @test.T) {
    #|  @distributions.seed(b"12345678223456783234567842345678")
    #|  let transpiler = Transpiler::new()
    $|\{content}
    $|  it.writeln(transpiler.to_string())
    $|  it.snapshot!(filename="nn/Tuple.py")
    $|}
  try {
    @generate.write_on_change!(path="../src/transpile/tuple_test.mbt", content~)
  } catch {
    error =>
      $|Failed to write content to src/transpile/tuple_test.mbt: \{Show::to_string(error)}
      |> println
  }
}
