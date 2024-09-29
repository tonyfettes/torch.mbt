pub struct Conv2d {
  weight : Array[Array[Tensor]]
  bias : Array[Tensor]
  padding : (Int, Int)
} derive(Show, ToJson, FromJson)

pub fn Conv2d::new[KernelSize : Shape2d](
  in_channels : Int,
  out_channels : Int,
  ~kernel_size : KernelSize,
  ~padding : Shape2d = 0
) -> Conv2d {
  if in_channels == 0 {
    abort("Input channels has zero width")
  }
  if out_channels == 0 {
    abort("Output channels has zero width")
  }
  let kernel_size = kernel_size.to_fixed_array()
  let weight_distribution : Continuous = {
    let mut n = in_channels
    for k in kernel_size {
      n *= k
    }
    let stdv = 1.0 / n.to_double().sqrt()
    Uniform::new(-stdv, stdv)
  }
  let weight = Array::makei(
    out_channels,
    fn(__) {
      Array::makei(
        in_channels,
        fn(__) { weight_distribution.sample(kernel_size) },
      )
    },
  )
  let bias = Array::makei(out_channels, fn(_i) { tensor(0.0) })
  Conv2d::{ weight, bias, padding: padding.to_tuple() }
}

pub fn Conv2d::output(self : Conv2d, logger : Logger) -> Unit {
  Show::output(self.weight, logger)
}

pub fn Conv2d::forward(self : Conv2d, input : Tensor) -> Tensor {
  if input.shape.length() != 4 {
    abort("Input tensor should be of shape (N, C, H, W)")
  }
  let out_channels = self.weight.length()
  let in_channels = self.weight[0].length()
  if input.shape[1] != in_channels {
    abort(
      "Input tensor has \{input.shape[1]} channels, which is different from \{self.weight.length()}",
    )
  }
  let batch_size = input.shape[0]
  let batch = FixedArray::make(batch_size, nan)
  for n = 0; n < batch_size; n = n + 1 {
    let channels = FixedArray::make(out_channels, nan)
    for oc = 0; oc < out_channels; oc = oc + 1 {
      let correlations = FixedArray::make(in_channels, nan)
      for ic = 0; ic < in_channels; ic = ic + 1 {
        let input = input[n][ic].to_tensor()
        let weight = self.weight[oc][ic]
        correlations[ic] = cross_correlation(
          input,
          weight,
          padding=self.padding.0,
        )
      }
      channels[oc] = Tensor::stack(correlations).sum(dim=0) + self.bias[oc]
    }
    batch[n] = Tensor::stack(channels)
  }
  Tensor::stack(batch)
}

pub fn Conv2d::parameters(self : Conv2d) -> Iter[Tensor] {
  self.weight.iter().flat_map(fn { channel => channel.iter() })
}

test "Conv2d - Multiple output channels" {
  seed(b"12345678223456783234567842345678")
  let image = tensor(FixedArray::makei(4 * 4, fn(i) { i.to_double() }))
  inspect!(
    image.reshape([4, 4]),
    content="[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]",
  )
  let image = image.reshape([1, 1, 4, 4])
  inspect!(image.shape, content="[1, 1, 4, 4]")
  let layer = Conv2d::new(1, 3, kernel_size=(2, 2))
  with_no_grad(
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
    },
  )
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
  seed(b"12345678223456783234567842345678")
  let image = tensor(FixedArray::makei(3 * 4 * 4, fn(i) { i.to_double() }))
  let image = image.reshape([1, 3, 4, 4])
  inspect!(
    image[0][0],
    content="[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]",
  )
  inspect!(
    image[0][1],
    content="[[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]",
  )
  inspect!(
    image[0][2],
    content="[[32, 33, 34, 35], [36, 37, 38, 39], [40, 41, 42, 43], [44, 45, 46, 47]]",
  )
  let layer = Conv2d::new(3, 1, kernel_size=(2, 2))
  with_no_grad(
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
    },
  )
  let output = layer.forward(image)
  inspect!(
    output[0][0],
    content="[[116, 122, 128], [140, 146, 152], [164, 170, 176]]",
  )
}

test "Conv2d - Backward" {
  seed(b"12345678223456783234567842345678")
  let image = tensor(FixedArray::makei(2 * 4 * 4, fn(i) { i.to_double() })).reshape(
    [1, 2, 4, 4],
  )
  inspect!(
    image[0][0],
    content="[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]",
  )
  inspect!(
    image[0][1],
    content="[[16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]",
  )
  let layer = Conv2d::new(2, 2, kernel_size=(2, 2))
  with_no_grad(
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
    },
  )
  let output = layer.forward(image)
  inspect!(output[0][0], content="[[47, 51, 55], [63, 67, 71], [79, 83, 87]]")
  inspect!(output[0][1], content="[[37, 41, 45], [53, 57, 61], [69, 73, 77]]")
  let loss = output.sum()
  inspect!(loss, content="1116")
  loss.backward()
  inspect!(layer.weight[0][0].grad, content="[45, 54, 81, 90]")
  inspect!(layer.weight[0][1].grad, content="[189, 198, 225, 234]")
  inspect!(layer.weight[1][0].grad, content="[45, 54, 81, 90]")
  inspect!(layer.weight[1][1].grad, content="[189, 198, 225, 234]")
}
