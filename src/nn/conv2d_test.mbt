test "Conv2d" {
  @distributions.seed(b"12345678223456783234567842345678")
  let image : @tensor.Tensor[Float] = @tensor.tensor(
    FixedArray::makei(4 * 4, fn(i) { i.to_double() }),
  )
  @json.inspect!(image.reshape([4, 4]), content=[
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
  ])
  let image = image.reshape([1, 1, 4, 4])
  let layer = @nn.Conv2d::new(1, 1, kernel_size=(2, 2))
  @tensor.with_no_grad(fn() {
    // kernel[0]:
    // 0 1
    // 0 1
    layer.weight[0][0][0][0] = 0.0
    layer.weight[0][0][0][1] = 1.0
    layer.weight[0][0][1][0] = 0.0
    layer.weight[0][0][1][1] = 1.0
    layer.bias[0] = 0.0
  })
  @json.inspect!(layer.weight[0][0], content=[[0, 1], [0, 1]])
  let output = layer.forward(image)
  @json.inspect!(output[0][0], content=[[6, 8, 10], [14, 16, 18], [22, 24, 26]])
}

test "Conv2d - Random" {
  @distributions.seed(b"12345678223456783234567842345678")
  let image = @distributions.Uniform::new(-1.0, 1.0).sample([1, 1, 4, 4])
  @json.inspect!(image[0][0], content=[
    [
      -0.2955702543258667, -0.8177422285079956, -0.8997843861579895, -0.1545487642288208,
    ],
    [
      -0.552005410194397, -0.0948251485824585, 0.6609057188034058, 0.21990251541137695,
    ],
    [
      0.43133544921875, 0.9582411050796509, 0.7228865623474121, -0.5188244581222534,
    ],
    [
      0.2701148986816406, -0.21887880563735962, 0.943917989730835, -0.4507488012313843,
    ],
  ])
  let layer = @nn.Conv2d::new(1, 1, kernel_size=(2, 2))
  @json.inspect!(layer.weight, content=[
    [
      [
        [-0.27374374866485596, 0.20872807502746582],
        [0.19746887683868408, 0.4987364411354065],
      ],
    ],
  ])
  @json.inspect!(layer.bias, content=[-0.45019540190696716])
  let output = layer.forward(image)
  @json.inspect!(output[0][0], content=[
    [-0.696267306804657, -0.1032610833644867, 0.004038006067276001],
    [0.2442050278186798, 0.2634645998477936, -0.7012234926223755],
    [-0.42408251762390137, -0.13407662510871887, -0.7947847247123718],
  ])
}

test "Conv2d - Multiple output channels" {
  @distributions.seed(b"12345678223456783234567842345678")
  let image : @tensor.Tensor[Float] = @tensor.tensor(
    FixedArray::makei(4 * 4, fn(i) { i.to_double() }),
  )
  inspect!(
    image.reshape([4, 4]),
    content="[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]",
  )
  let image = image.reshape([1, 1, 4, 4])
  inspect!(image.shape, content="[1, 1, 4, 4]")
  let layer = @nn.Conv2d::new(1, 3, kernel_size=(2, 2))
  @tensor.with_no_grad(fn() {
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
    layer.bias[0] = 0.0
    layer.bias[1] = 0.0
    layer.bias[2] = 0.0
  })
  inspect!(layer.weight[0][0], content="[[0, 1], [0, 1]]")
  inspect!(layer.weight[1][0], content="[[0, 0], [1, 1]]")
  inspect!(layer.weight[2][0], content="[[1, 0], [0, 1]]")
  let output = layer.forward(image)
  inspect!(output.shape, content="[1, 3, 3, 3]")
  inspect!(output[0][0], content="[[6, 8, 10], [14, 16, 18], [22, 24, 26]]")
  inspect!(output[0][1], content="[[9, 11, 13], [17, 19, 21], [25, 27, 29]]")
  inspect!(output[0][2], content="[[5, 7, 9], [13, 15, 17], [21, 23, 25]]")
}

test "Conv2d - Multiple input channels" {
  @distributions.seed(b"12345678223456783234567842345678")
  let image : @tensor.Tensor[Float] = @tensor.tensor(
    FixedArray::makei(3 * 4 * 4, fn(i) { i.to_double() }),
  )
  let image = image.reshape([1, 3, 4, 4])
  @json.inspect!(image[0][0], content=[
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
  ])
  @json.inspect!(image[0][1], content=[
    [16, 17, 18, 19],
    [20, 21, 22, 23],
    [24, 25, 26, 27],
    [28, 29, 30, 31],
  ])
  @json.inspect!(image[0][2], content=[
    [32, 33, 34, 35],
    [36, 37, 38, 39],
    [40, 41, 42, 43],
    [44, 45, 46, 47],
  ])
  let layer = @nn.Conv2d::new(3, 1, kernel_size=(2, 2))
  @tensor.with_no_grad(fn() {
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
    layer.bias[0] = 0.0
  })
  let output = layer.forward(image)
  inspect!(
    output[0][0],
    content="[[116, 122, 128], [140, 146, 152], [164, 170, 176]]",
  )
}

test "Conv2d - Padding" {
  let image : @tensor.Tensor[Float] = @tensor.tensor(
    FixedArray::makei(4 * 4, fn(i) { i.to_double() }),
  ).reshape([1, 1, 4, 4])
  let layer = @nn.Conv2d::new(1, 1, kernel_size=(2, 2), padding=(1, 1))
  @tensor.with_no_grad(fn() {
    // kernel:
    // 0 1
    // 0 1
    layer.weight[0][0][0][0] = 0.0
    layer.weight[0][0][0][1] = 1.0
    layer.weight[0][0][1][0] = 0.0
    layer.weight[0][0][1][1] = 1.0
    layer.bias[0] = 0.0
  })
  let output = layer.forward(image)
  @json.inspect!(output[0][0], content=[
    [0, 1, 2, 3, 0],
    [4, 6, 8, 10, 0],
    [12, 14, 16, 18, 0],
    [20, 22, 24, 26, 0],
    [12, 13, 14, 15, 0],
  ])
}

test "Conv2d - Backward" {
  @distributions.seed(b"12345678223456783234567842345678")
  let image : @tensor.Tensor[Float] = @tensor.tensor(
    FixedArray::makei(2 * 4 * 4, fn(i) { i.to_double() }),
  ).reshape([1, 2, 4, 4])
  @json.inspect!(image[0][0], content=[
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
  ])
  @json.inspect!(image[0][1], content=[
    [16, 17, 18, 19],
    [20, 21, 22, 23],
    [24, 25, 26, 27],
    [28, 29, 30, 31],
  ])
  let layer = @nn.Conv2d::new(2, 2, kernel_size=2, padding=2)
  @tensor.with_no_grad(fn() {
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
    layer.bias[0] = 0.0
    layer.bias[1] = 0.0
  })
  let output = layer.forward(image)
  @json.inspect!(output[0][0], content=[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 16, 34, 37, 40, 19, 0],
    [0, 24, 47, 51, 55, 23, 0],
    [0, 36, 63, 67, 71, 27, 0],
    [0, 48, 79, 83, 87, 31, 0],
    [0, 12, 13, 14, 15, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
  ])
  @json.inspect!(output[0][1], content=[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 3, 0],
    [0, 16, 37, 41, 45, 29, 0],
    [0, 20, 53, 57, 61, 41, 0],
    [0, 24, 69, 73, 77, 53, 0],
    [0, 28, 69, 72, 75, 46, 0],
    [0, 0, 0, 0, 0, 0, 0],
  ])
  let loss = output.sum()
  let optimizer = @optim.SGD::new(layer.parameters(), learning_rate=0.1)
  @json.inspect!(loss, content=1984)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  @json.inspect!(layer.weight[0][0], content=[[-12, -11], [-12, -11]])
  @json.inspect!(layer.weight[0][1], content=[
    [-37.60000228881836, -37.60000228881836],
    [-36.60000228881836, -36.60000228881836],
  ])
  @json.inspect!(layer.weight[1][0], content=[[-11, -12], [-11, -12]])
  @json.inspect!(layer.weight[1][1], content=[
    [-36.60000228881836, -36.60000228881836],
    [-37.60000228881836, -37.60000228881836],
  ])
}

test "Conv2d - train average kernel" {
  @distributions.seed(b"12345678223456783234567842345678")
  let model = @nn.Conv2d::new(1, 1, kernel_size=(2, 2))
  @json.inspect!(model.weight[0][0], content=[
    [-0.14778512716293335, -0.4088711142539978],
    [-0.44989219307899475, -0.0772743821144104],
  ])
  @json.inspect!(model.bias[0], content=-0.2760027050971985)
  let refer = @nn.AvgPool2d::new(2, stride=1)
  let optimizer = @optim.SGD::new(model.parameters(), learning_rate=0.001)
  let criterion = @nn.MSELoss::new()
  for i = 0; i < 10; i = i + 1 {
    let input = @distributions.Uniform::new(-1.0, 1.0).sample([1, 1, 28, 28])
    let output = model.forward(input)
    let target = refer.forward(input)
    let loss = criterion.forward(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  }
  @json.inspect!(model.weight, content=[
    [
      [
        [-0.1450861096382141, -0.40445324778556824],
        [-0.4451870620250702, -0.07506425678730011],
      ],
    ],
  ])
}

test "Conv2d - train random kernel" {
  @distributions.seed(b"12345678223456783234567842345678")
  let model = @nn.Conv2d::new(1, 1, kernel_size=2)
  let refer = @nn.Conv2d::new(1, 1, kernel_size=2)
  let optimizer = @optim.SGD::new(model.parameters(), learning_rate=0.001)
  @tensor.with_no_grad_error!(fn() {
    let distrib = @distributions.Uniform::new(-8.0, 8.0)
    let kernel = distrib.sample([2, 2])
    @json.inspect!(kernel, content=[
      [5.783092498779297, -4.150595664978027],
      [2.160919189453125, -1.751030445098877],
    ])
    refer.weight[0][0] = kernel
  })
  let loss = @nn.MSELoss::new()
  for i = 0; i < 10; i = i + 1 {
    let input = @distributions.Uniform::new(-1.0, 1.0).sample([1, 1, 28, 28])
    let output = model.forward(input)
    let target = refer.forward(input)
    let loss = loss.forward(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  }
  @json.inspect!(model.weight, content=[
    [
      [
        [-0.107294961810112, -0.43479594588279724],
        [-0.43136852979660034, -0.08917967975139618],
      ],
    ],
  ])
}

test "Conv2d - Sigmoid - AvgPool2d" {
  @distributions.seed(b"12345678223456783234567842345678")
  let image = @distributions.Uniform::new(-1.0, 1.0).sample([1, 1, 4, 4])
  @json.inspect!(image[0][0], content=[
    [
      -0.2955702543258667, -0.8177422285079956, -0.8997843861579895, -0.1545487642288208,
    ],
    [
      -0.552005410194397, -0.0948251485824585, 0.6609057188034058, 0.21990251541137695,
    ],
    [
      0.43133544921875, 0.9582411050796509, 0.7228865623474121, -0.5188244581222534,
    ],
    [
      0.2701148986816406, -0.21887880563735962, 0.943917989730835, -0.4507488012313843,
    ],
  ])
  let model = (
    @nn.Conv2d::new(1, 2, kernel_size=2, padding=1),
    @nn.Sigmoid::new(),
    @nn.AvgPool2d::new(2, stride=2),
  )
  @json.inspect!(model.0.weight, content=[
    [
      [
        [-0.27374374866485596, 0.20872807502746582],
        [0.19746887683868408, 0.4987364411354065],
      ],
    ],
    [
      [
        [-0.45019540190696716, 0.09324383735656738],
        [-0.3585057854652405, -0.1407204568386078],
      ],
    ],
  ])
  @json.inspect!(model.0.bias, content=[
    -0.012118875980377197, 0.11915922164916992,
  ])
  let optimizer = @optim.SGD::new(
    @nn.Module::parameters(model),
    learning_rate=0.1,
  )
  let output = @nn.Module::forward(model, image)
  @json.inspect!(output.shape, content=[1, 2, 2, 2])
  @json.inspect!(output[0], content=[
    [
      [0.423063725233078, 0.49365341663360596],
      [0.5606590509414673, 0.5225741267204285],
    ],
    [
      [0.5655426383018494, 0.5977144241333008],
      [0.5091142654418945, 0.41399282217025757],
    ],
  ])
  let target : @tensor.Tensor[Float] = @tensor.tensor([
    0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
  ]).reshape([1, 2, 2, 2])
  let criterion = @nn.CrossEntropyLoss::new()
  let loss = criterion.forward(output, target)
  @json.inspect!(loss, content=0.18322280049324036)
  loss.backward()
  optimizer.step()
  @json.inspect!(model.0.weight, content=[
    [
      [
        [-0.2743982970714569, 0.2080078125],
        [0.19675858318805695, 0.49815377593040466],
      ],
    ],
    [
      [
        [-0.4495529532432556, 0.09396219253540039],
        [-0.3577914535999298, -0.14009231328964233],
      ],
    ],
  ])
  @json.inspect!(model.0.bias, content=[
    -0.012118875980377197, 0.11915922164916992,
  ])
}

test "MNIST" {
  @distributions.seed(b"12345678223456783234567842345678")
  let label = 5
  let image : @tensor.Tensor[Float] = @tensor.tensor(
    [
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
    ].map(fn(i) { i.to_float() / 255.0 }),
  ).reshape([1, 1, 28, 28])
  let tuple = (
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
  )
  let model = tuple as &@nn.Module
  let optimizer = @optim.SGD::new(model.parameters(), learning_rate=0.1)
  let criterion = @nn.CrossEntropyLoss::new()
  let output = model.forward(image)
  @json.inspect!(output[0][label], content=-0.3357795774936676)
  let target = FixedArray::make(10, 0.0)
  target[label] = 1.0
  let loss = criterion.forward(output, @tensor.tensor([target]))
  @json.inspect!(loss, content=2.6749267578125)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  let output = model.forward(image)
  inspect!(output[0][label], content="1.8095556497573853")
}
