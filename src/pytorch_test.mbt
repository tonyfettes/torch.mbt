test (it : @test.T) {
  @torch.seed(b"12345678223456783234567842345678")
  let model = @torch.Sequential::new(
    [
      @torch.Linear::new(1, 2),
      @torch.ReLU::new(),
      @torch.Linear::new(2, 2),
      @torch.ReLU::new(),
      @torch.Linear::new(2, 1),
      @torch.Softmax::new(),
    ],
  )
  it.write(@torch.transpile(model, "model"))
  it.snapshot!(filename="ToPyTorchSource/Linear.py")
}

test (it : @test.T) {
  @torch.seed(b"12345678223456783234567842345678")
  let model = @torch.Sequential::new(
    [
      @torch.Conv2d::new(1, 6, kernel_size=5, padding=2),
      @torch.Sigmoid::new(),
      @torch.AvgPool2d::new(2, stride=2),
      @torch.Conv2d::new(6, 16, kernel_size=5),
      @torch.Sigmoid::new(),
      @torch.AvgPool2d::new(2, stride=2),
      @torch.Flatten::new(),
      @torch.Linear::new(400, 120),
      @torch.Sigmoid::new(),
      @torch.Linear::new(120, 84),
      @torch.Sigmoid::new(),
      @torch.Linear::new(84, 10),
    ],
  )
  it.write(@torch.transpile(model, "model"))
  it.snapshot!(filename="ToPyTorchSource/MNIST/Sequential.py")
}

test (it : @test.T) {
  @torch.seed(b"12345678223456783234567842345678")
  let model = (
    @torch.Conv2d::new(1, 6, kernel_size=5, padding=2),
    @torch.Sigmoid::new(),
    @torch.AvgPool2d::new(2, stride=2),
    @torch.Conv2d::new(6, 16, kernel_size=5),
    @torch.Sigmoid::new(),
    @torch.AvgPool2d::new(2, stride=2),
    @torch.Flatten::new(),
    @torch.Linear::new(400, 120),
    @torch.Sigmoid::new(),
    @torch.Linear::new(120, 84),
    @torch.Sigmoid::new(),
    @torch.Linear::new(84, 10),
  )
  it.write(@torch.transpile(model, "model"))
  it.snapshot!(filename="ToPyTorchSource/MNIST/Tuple.py")
}
