test "Conv2d" {
  @torch.seed(b"12345678223456783234567842345678")
  let image = @torch.tensor(FixedArray::makei(4 * 4, fn(i) { i.to_double() }))
  @json.inspect!(
    image.reshape([4, 4]),
    content=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
  )
  let image = image.reshape([1, 1, 4, 4])
  let layer = @torch.Conv2d::new(1, 1, kernel_size=(2, 2))
  @torch.with_no_grad(
    fn() {
      // kernel[0]:
      // 0 1
      // 0 1
      layer.weight[0][0][0][0] = 0.0
      layer.weight[0][0][0][1] = 1.0
      layer.weight[0][0][1][0] = 0.0
      layer.weight[0][0][1][1] = 1.0
      layer.bias[0] = @torch.tensor(0.0)
    },
  )
  @json.inspect!(layer.weight[0][0], content=[[0, 1], [0, 1]])
  let output = layer.forward(image)
  @json.inspect!(output[0][0], content=[[6, 8, 10], [14, 16, 18], [22, 24, 26]])
}

test "Conv2d - Random" {
  @torch.seed(b"12345678223456783234567842345678")
  let image = @torch.Uniform::new(-1.0, 1.0).sample([1, 1, 4, 4])
  @json.inspect!(
    image[0][0],
    content=[
      [
        -0.29557028390369, -0.8177422144361883, -0.8997844018869048, -0.1545487762608706,
      ],
      [
        -0.5520054503667071, -0.09482517741472507, 0.6609056774831252, 0.21990249887250046,
      ],
      [
        0.4313355035432884, 0.9582411244732583, 0.7228865545307872, -0.5188244284458443,
      ],
      [
        0.2701149512079253, -0.21887879807607646, 0.9439179395000068, -0.4507487893925779,
      ],
    ],
  )
  let layer = @torch.Conv2d::new(1, 1, kernel_size=(2, 2))
  @json.inspect!(
    layer.weight,
    content=[
      [
        [
          [-0.27374372773124167, 0.20872805188361476],
          [0.1974688954407423, 0.49873644685786933],
        ],
      ],
    ],
  )
  @json.inspect!(layer.bias, content=[-0.4501954011239082])
  let output = layer.forward(image)
  @json.inspect!(
    output[0][0],
    content=[
      [-0.6962673077814596, -0.10326111808999416, 0.004037975279056305],
      [0.24420507313809514, 0.26346463906788964, -0.7012234072968717],
      [-0.42408251960853705, -0.1340766715623704, -0.7947846902544631],
    ],
  )
}

test "Conv2d - Multiple output channels" {
  @torch.seed(b"12345678223456783234567842345678")
  let image = @torch.tensor(FixedArray::makei(4 * 4, fn(i) { i.to_double() }))
  inspect!(
    image.reshape([4, 4]),
    content="[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]",
  )
  let image = image.reshape([1, 1, 4, 4])
  inspect!(image.shape(), content="[1, 1, 4, 4]")
  let layer = @torch.Conv2d::new(1, 3, kernel_size=(2, 2))
  @torch.with_no_grad(
    fn() {
      // kernel[0]:
      // 0 1
      // 0 1
      layer.weight[0][0][0][0] = 0.0
      layer.weight[0][0][0][1] = 1.0
      layer.weight[0][0][1][0] = 0.0
      layer.weight[0][0][1][1] = 1.0
      // kernel[1]:
      // 0 0
      // 1 1
      layer.weight[1][0][0][0] = 0.0
      layer.weight[1][0][0][1] = 0.0
      layer.weight[1][0][1][0] = 1.0
      layer.weight[1][0][1][1] = 1.0
      // kernel[2]:
      // 1 0
      // 0 1
      layer.weight[2][0][0][0] = 1.0
      layer.weight[2][0][0][1] = 0.0
      layer.weight[2][0][1][0] = 0.0
      layer.weight[2][0][1][1] = 1.0
      layer.bias[0] = @torch.tensor(0.0)
      layer.bias[1] = @torch.tensor(0.0)
      layer.bias[2] = @torch.tensor(0.0)
    },
  )
  inspect!(layer.weight[0][0], content="[[0, 1], [0, 1]]")
  inspect!(layer.weight[1][0], content="[[0, 0], [1, 1]]")
  inspect!(layer.weight[2][0], content="[[1, 0], [0, 1]]")
  let output = layer.forward(image)
  inspect!(output.shape(), content="[1, 3, 3, 3]")
  inspect!(output[0][0], content="[[6, 8, 10], [14, 16, 18], [22, 24, 26]]")
  inspect!(output[0][1], content="[[9, 11, 13], [17, 19, 21], [25, 27, 29]]")
  inspect!(output[0][2], content="[[5, 7, 9], [13, 15, 17], [21, 23, 25]]")
}

test "Conv2d - Multiple input channels" {
  @torch.seed(b"12345678223456783234567842345678")
  let image = @torch.tensor(
    FixedArray::makei(3 * 4 * 4, fn(i) { i.to_double() }),
  )
  let image = image.reshape([1, 3, 4, 4])
  @json.inspect!(
    image[0][0],
    content=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
  )
  @json.inspect!(
    image[0][1],
    content=[
      [16, 17, 18, 19],
      [20, 21, 22, 23],
      [24, 25, 26, 27],
      [28, 29, 30, 31],
    ],
  )
  @json.inspect!(
    image[0][2],
    content=[
      [32, 33, 34, 35],
      [36, 37, 38, 39],
      [40, 41, 42, 43],
      [44, 45, 46, 47],
    ],
  )
  let layer = @torch.Conv2d::new(3, 1, kernel_size=(2, 2))
  @torch.with_no_grad(
    fn() {
      // kernel[0]:
      // 0 1
      // 0 1
      layer.weight[0][0][0][0] = 0.0
      layer.weight[0][0][0][1] = 1.0
      layer.weight[0][0][1][0] = 0.0
      layer.weight[0][0][1][1] = 1.0
      // kernel[1]:
      // 0 0
      // 1 1
      layer.weight[0][1][0][0] = 0.0
      layer.weight[0][1][0][1] = 0.0
      layer.weight[0][1][1][0] = 1.0
      layer.weight[0][1][1][1] = 1.0
      // kernel[2]:
      // 1 0
      // 0 1
      layer.weight[0][2][0][0] = 1.0
      layer.weight[0][2][0][1] = 0.0
      layer.weight[0][2][1][0] = 0.0
      layer.weight[0][2][1][1] = 1.0
      layer.bias[0] = @torch.tensor(0.0)
    },
  )
  let output = layer.forward(image)
  inspect!(
    output[0][0],
    content="[[116, 122, 128], [140, 146, 152], [164, 170, 176]]",
  )
}

test "Conv2d - Padding" {
  let image = @torch.tensor(FixedArray::makei(4 * 4, fn(i) { i.to_double() })).reshape(
    [1, 1, 4, 4],
  )
  let layer = @torch.Conv2d::new(1, 1, kernel_size=(2, 2), padding=(1, 1))
  @torch.with_no_grad(
    fn() {
      // kernel:
      // 0 1
      // 0 1
      layer.weight[0][0][0][0] = 0.0
      layer.weight[0][0][0][1] = 1.0
      layer.weight[0][0][1][0] = 0.0
      layer.weight[0][0][1][1] = 1.0
      layer.bias[0] = @torch.tensor(0.0)
    },
  )
  let output = layer.forward(image)
  @json.inspect!(
    output[0][0],
    content=[
      [0, 1, 2, 3, 0],
      [4, 6, 8, 10, 0],
      [12, 14, 16, 18, 0],
      [20, 22, 24, 26, 0],
      [12, 13, 14, 15, 0],
    ],
  )
}

test "Conv2d - Backward" {
  @torch.seed(b"12345678223456783234567842345678")
  let image = @torch.tensor(
    FixedArray::makei(2 * 4 * 4, fn(i) { i.to_double() }),
  ).reshape([1, 2, 4, 4])
  @json.inspect!(
    image[0][0],
    content=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
  )
  @json.inspect!(
    image[0][1],
    content=[
      [16, 17, 18, 19],
      [20, 21, 22, 23],
      [24, 25, 26, 27],
      [28, 29, 30, 31],
    ],
  )
  let layer = @torch.Conv2d::new(2, 2, kernel_size=2, padding=2)
  @torch.with_no_grad(
    fn() {
      // 0 1
      // 0 1
      layer.weight[0][0][0][0] = 0.0
      layer.weight[0][0][0][1] = 1.0
      layer.weight[0][0][1][0] = 0.0
      layer.weight[0][0][1][1] = 1.0
      // 0 0
      // 1 1
      layer.weight[0][1][0][0] = 0.0
      layer.weight[0][1][0][1] = 0.0
      layer.weight[0][1][1][0] = 1.0
      layer.weight[0][1][1][1] = 1.0
      // 1 0
      // 1 0
      layer.weight[1][0][0][0] = 1.0
      layer.weight[1][0][0][1] = 0.0
      layer.weight[1][0][1][0] = 1.0
      layer.weight[1][0][1][1] = 0.0
      // 1 1
      // 0 0
      layer.weight[1][1][0][0] = 1.0
      layer.weight[1][1][0][1] = 1.0
      layer.weight[1][1][1][0] = 0.0
      layer.weight[1][1][1][1] = 0.0
      layer.bias[0] = @torch.tensor(0.0)
      layer.bias[1] = @torch.tensor(0.0)
    },
  )
  let output = layer.forward(image)
  @json.inspect!(
    output[0][0],
    content=[
      [0, 0, 0, 0, 0, 0, 0],
      [0, 16, 34, 37, 40, 19, 0],
      [0, 24, 47, 51, 55, 23, 0],
      [0, 36, 63, 67, 71, 27, 0],
      [0, 48, 79, 83, 87, 31, 0],
      [0, 12, 13, 14, 15, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
    ],
  )
  @json.inspect!(
    output[0][1],
    content=[
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 2, 3, 0],
      [0, 16, 37, 41, 45, 29, 0],
      [0, 20, 53, 57, 61, 41, 0],
      [0, 24, 69, 73, 77, 53, 0],
      [0, 28, 69, 72, 75, 46, 0],
      [0, 0, 0, 0, 0, 0, 0],
    ],
  )
  let loss = output.sum()
  let optimizer = @torch.SGD::new(layer.parameters(), learning_rate=0.1)
  @json.inspect!(loss, content=1984)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  @json.inspect!(layer.weight[0][0], content=[[-12, -11], [-12, -11]])
  @json.inspect!(layer.weight[0][1], content=[[-37.6, -37.6], [-36.6, -36.6]])
  @json.inspect!(layer.weight[1][0], content=[[-11, -12], [-11, -12]])
  @json.inspect!(layer.weight[1][1], content=[[-36.6, -36.6], [-37.6, -37.6]])
}

test "Conv2d - train average kernel" {
  @torch.seed(b"12345678223456783234567842345678")
  let model = @torch.Conv2d::new(1, 1, kernel_size=(2, 2))
  @json.inspect!(
    model.weight[0][0],
    content=[
      [-0.147785141951845, -0.40887110721809417],
      [-0.4498922009434524, -0.0772743881304353],
    ],
  )
  @json.inspect!(model.bias[0], content=-0.27600272518335356)
  let refer = @torch.AvgPool2d::new(2, stride=1)
  let optimizer = @torch.SGD::new(model.parameters(), learning_rate=0.001)
  let criterion = @torch.MSELoss::new()
  for i = 0; i < 10; i = i + 1 {
    let input = @torch.Uniform::new(-1, 1).sample([1, 1, 28, 28])
    let output = model.forward(input)
    let target = refer.forward(input)
    let loss = criterion.forward(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  }
  @json.inspect!(
    model.weight,
    content=[
      [
        [
          [-0.1450861159407917, -0.40445321042353977],
          [-0.4451870982662845, -0.07506427810887198],
        ],
      ],
    ],
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
      @json.inspect!(
        kernel,
        content=[
          [5.783092436246298, -4.1505954275667545],
          [2.1609196096634022, -1.7510303846086117],
        ],
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
  @json.inspect!(
    model.weight,
    content=[
      [
        [
          [-0.10729496084550386, -0.43479597964267996],
          [-0.431368564979322, -0.08917967937604632],
        ],
      ],
    ],
  )
}

test "Conv2d - Sigmoid - AvgPool2d" {
  @torch.seed(b"12345678223456783234567842345678")
  let image = @torch.Uniform::new(-1, 1).sample([1, 1, 4, 4])
  @json.inspect!(
    image[0][0],
    content=[
      [
        -0.29557028390369, -0.8177422144361883, -0.8997844018869048, -0.1545487762608706,
      ],
      [
        -0.5520054503667071, -0.09482517741472507, 0.6609056774831252, 0.21990249887250046,
      ],
      [
        0.4313355035432884, 0.9582411244732583, 0.7228865545307872, -0.5188244284458443,
      ],
      [
        0.2701149512079253, -0.21887879807607646, 0.9439179395000068, -0.4507487893925779,
      ],
    ],
  )
  let model = (
    @torch.Conv2d::new(1, 2, kernel_size=2, padding=1),
    @torch.Sigmoid::new(),
    @torch.AvgPool2d::new(2, stride=2),
  )
  @json.inspect!(
    model.0.weight,
    content=[
      [
        [
          [-0.27374372773124167, 0.20872805188361476],
          [0.1974688954407423, 0.49873644685786933],
        ],
      ],
      [
        [
          [-0.4501954011239082, 0.0932438253080703],
          [-0.35850578001958444, -0.14072045572696013],
        ],
      ],
    ],
  )
  @json.inspect!(
    model.0.bias,
    content=[-0.012118885501261834, 0.1191592062303477],
  )
  let optimizer = @torch.SGD::new(
    @torch.Module::parameters(model),
    learning_rate=0.1,
  )
  let output = @torch.Module::forward(model, image)
  @json.inspect!(output.shape(), content=[1, 2, 2, 2])
  @json.inspect!(
    output[0],
    content=[
      [
        [0.42306371266340065, 0.49365342570168114],
        [0.560659050739706, 0.522574086097089],
      ],
      [
        [0.5655426357800895, 0.5977144269072799],
        [0.509114214887437, 0.41399279658010235],
      ],
    ],
  )
  let target = @torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]).reshape(
    [1, 2, 2, 2],
  )
  let criterion = @torch.CrossEntropyLoss::new()
  let loss = criterion.forward(output, target)
  @json.inspect!(loss, content=0.18322279098221156)
  loss.backward()
  optimizer.step()
  @json.inspect!(
    model.0.weight,
    content=[
      [
        [
          [-0.274398280939688, 0.2080077914728873],
          [0.19675859698058834, 0.4981537785986445],
        ],
      ],
      [
        [
          [-0.44955294619094094, 0.09396218030855882],
          [-0.3577914617572429, -0.14009231886256904],
        ],
      ],
    ],
  )
  @json.inspect!(
    model.0.bias,
    content=[-0.012118885501261834, 0.1191592062303477],
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
    ],
  )
  let optimizer = @torch.SGD::new(model.parameters(), learning_rate=0.1)
  let criterion = @torch.CrossEntropyLoss::new()
  let output = model.forward(image)
  @json.inspect!(output[0][label], content=-0.3365644644720264)
  let target = FixedArray::make(10, 0.0)
  target[label] = 1.0
  let loss = criterion.forward(output, @torch.tensor([target]))
  @json.inspect!(loss, content=2.675598250228825)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  let output = model.forward(image)
  @json.inspect!(output[0][label], content=1.8093816993511227)
}
