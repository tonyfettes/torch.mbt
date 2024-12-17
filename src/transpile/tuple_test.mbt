test (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let transpiler = Transpiler::new()
  let model = (@nn.Linear::new(1, 1), @nn.Linear::new(1, 1))
  Transpile::transpile(model, transpiler, "tuple2")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple3")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple4")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple5")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple6")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple7")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple8")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple9")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple10")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple11")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple12")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple13")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple14")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple15")
  let model = (
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
    @nn.Linear::new(1, 1),
  )
  Transpile::transpile(model, transpiler, "tuple16")
  it.writeln(transpiler.to_string())
  it.snapshot!(filename="nn/Tuple.py")
}
