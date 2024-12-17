///|
trait Module: @nn.Module + @transpile.Transpile {}

test (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let model : @nn.Sequential[&Module] = @nn.Sequential::new([
    @nn.Linear::new(1, 2),
    @nn.ReLU::new(),
    @nn.Linear::new(2, 2),
    @nn.ReLU::new(),
    @nn.Linear::new(2, 1),
    @nn.Softmax::new(),
  ])
  it.writeln(@transpile.transpile(model, "model"))
  it.snapshot!(filename="nn/Linear.py")
}

test (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let model : @nn.Sequential[&Module] = @nn.Sequential::new([
    @nn.Conv2d::new(1, 6, kernel_size=5, padding=2),
    @nn.Sigmoid::new(),
    @nn.AvgPool2d::new(2, stride=2),
    @nn.Conv2d::new(6, 16, kernel_size=5),
    @nn.Sigmoid::new(),
    @nn.AvgPool2d::new(2, stride=2),
    @nn.Flatten::new(),
    @nn.Linear::new(400, 120),
    @nn.Sigmoid::new(),
    @nn.Linear::new(120, 84),
    @nn.Sigmoid::new(),
    @nn.Linear::new(84, 10),
  ])
  it.writeln(@transpile.transpile(model, "model"))
  it.snapshot!(filename="nn/Sequential.py")
}

// test (it : @test.T) {
//   @distributions.seed(b"12345678223456783234567842345678")
//   let model = (
//     @nn.Conv2d::new(1, 6, kernel_size=5, padding=2),
//     @nn.Sigmoid::new(),
//     @nn.AvgPool2d::new(2, stride=2),
//     @nn.Conv2d::new(6, 16, kernel_size=5),
//     @nn.Sigmoid::new(),
//     @nn.AvgPool2d::new(2, stride=2),
//     @nn.Flatten::new(),
//     @nn.Linear::new(400, 120),
//     @nn.Sigmoid::new(),
//     @nn.Linear::new(120, 84),
//     @nn.Sigmoid::new(),
//     @nn.Linear::new(84, 10),
//   )
//   it.writeln(@transpile.transpile(model, "model"))
//   it.snapshot!(filename="nn/Tuple.py")
// }
