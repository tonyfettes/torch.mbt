test "Conv2d - train average kernel" {
  @torch.seed(b"12345678223456783234567842345678")
  let model = @torch.Conv2d::new(1, 1, kernel_size=(2, 2))
  let refer = @torch.AvgPool2d::new(2, stride=1)
  let optimizer = @torch.SGD::new(model.parameters(), learning_rate=0.001)
  for i = 0; i < 10; i = i + 1 {
    let input = @torch.Uniform::new(-1, 1).sample([1, 1, 28, 28])
    let output = model.forward(input)
    let target = refer.forward(input)
    let loss = @torch.MSELoss::new()
    let loss = loss.forward(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  }
  inspect!(
    model.weight,
    content="[[[[0.2398214365174994, 0.23837034340298877], [0.2372611163600896, 0.237249868299508]]]]",
  )
}

test "Conv2d - train random kernel" {
  @torch.seed(b"12345678223456783234567842345678")
  let model = @torch.Conv2d::new(1, 1, kernel_size=2)
  let refer = @torch.Conv2d::new(1, 1, kernel_size=2)
  let optimizer = @torch.SGD::new(model.parameters(), learning_rate=0.001)
  @torch.with_no_grad_error!(
    fn() {
      let distrib = @torch.Uniform::new(-8, 8)
      let kernel = distrib.sample([2, 2])
      inspect!(
        kernel,
        content="[[5.783092436246298, -4.1505954275667545], [2.1609196096634022, -1.7510303846086117]]",
      )
      refer.weight[0][0] = kernel
    },
  )
  let loss = @torch.MSELoss::new()
  for i = 0; i < 10; i = i + 1 {
    let input = @torch.Uniform::new(-1, 1).sample([1, 1, 28, 28])
    let output = model.forward(input)
    let target = refer.forward(input)
    let loss = loss.forward(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  }
  inspect!(
    model.weight,
    content="[[[[5.734003024208306, -4.1928148324930925], [2.115193989982714, -1.7950790294008958]]]]",
  )
}

test "Conv2d - Sigmoid - AvgPool2d" {
  @torch.seed(b"12345678223456783234567842345678")
  let image = @torch.tensor(
    FixedArray::makei(1 * 4 * 4, fn(i) { i.to_double() }),
  ).reshape([1, 1, 4, 4])
  let model = @torch.Sequential::new(
    [
      @torch.Conv2d::new(1, 2, kernel_size=(2, 2), padding=(1, 1)),
      @torch.AvgPool2d::new(2, stride=2),
      @torch.Conv2d::new(2, 4, kernel_size=(2, 2), padding=(1, 1)),
      @torch.AvgPool2d::new(2, stride=2),
      @torch.Flatten::new(),
    ],
  )
  let optimizer = @torch.SGD::new(model.parameters(), learning_rate=0.1)
  let output = model.forward(image)
  let output = output[0].to_tensor()
  inspect!(output.shape, content="[4]")
  inspect!(
    output,
    content="[0.4052299802353421, -1.2087408488805544, 0.2599913853969742, -0.08152780176751948]",
  )
  let target = @torch.tensor([0.0, 1.0, 0.0, 0.0])
  let loss = @torch.MSELoss::new()
  let loss = loss.forward(output, target)
  inspect!(loss, content="1.2792474443342026")
  loss.backward()
  optimizer.step()
  let output = model.forward(image)
  inspect!(
    output,
    content="[[0.0055807694974617725, 0.6621095254158275, 0.0995642683094404, -0.05257892536220428]]",
  )
}

test "MNIST" {
  @torch.seed(b"12345678223456783234567842345678")
  let label = 5
  let image = @torch.tensor(
    [
      0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172,
      253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253,
      253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11,
      0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1,
      154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240,
      253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253,
      253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148,
      229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253,
      195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253,
      253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
  ).reshape([1, 1, 28, 28])
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
      @torch.Softmax::new(),
    ],
  )
  let optimizer = @torch.SGD::new(model.parameters(), learning_rate=0.1)
  let loss = @torch.CrossEntropyLoss::new()
  let output = model.forward(image)
  inspect!(output[0][label], content="0.06886561719734109")
  let target = FixedArray::make(10, 0.0)
  target[label] = 1.0
  let loss = loss.forward(output, @torch.tensor(target))
  inspect!(loss, content="2.675598250228825")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  let output = model.forward(image)
  inspect!(output[0][label], content="0.44713147154564087")
}
