pub(readonly) struct Conv2d {
  weight : Array[Array[@tensor.Tensor]]
  bias : Array[@tensor.Tensor]
  padding : (Int, Int)
} derive(Show, ToJson, FromJson)

pub fn Conv2d::new[KernelSize : @shape.Shape2d](
  in_channels : Int,
  out_channels : Int,
  kernel_size~ : KernelSize,
  padding~ : @shape.Shape2d = 0
) -> Conv2d {
  if in_channels == 0 {
    abort("Input channels has zero width")
  }
  if out_channels == 0 {
    abort("Output channels has zero width")
  }
  let kernel_size = kernel_size.to_fixed_array()
  let weight_distribution : @distributions.Continuous = {
    let mut n = in_channels
    for k in kernel_size {
      n *= k
    }
    let stdv = (1.0 : Float) / n.to_float().sqrt()
    @distributions.Uniform::new(-stdv, stdv)
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
  let bias = Array::makei(
    out_channels,
    fn(__) { weight_distribution.sample([]) },
  )
  Conv2d::{ weight, bias, padding: padding.to_tuple() }
}

pub fn Conv2d::output(self : Conv2d, logger : Logger) -> Unit {
  Show::output(self.weight, logger)
}

pub fn Conv2d::forward(self : Conv2d, input : @tensor.Tensor) -> @tensor.Tensor {
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
  let batch = FixedArray::makei(
    batch_size,
    fn(n) {
      let channels = FixedArray::makei(
        out_channels,
        fn(oc) {
          let correlations = FixedArray::makei(
            in_channels,
            fn(ic) {
              let input = input[n][ic].to_tensor()
              let weight = self.weight[oc][ic]
              @tensor.correlate2d(input, weight, padding=self.padding.0)
            },
          )
          @tensor.Tensor::stack(correlations).sum(dim=0) + self.bias[oc]
        },
      )
      @tensor.Tensor::stack(channels)
    },
  )
  @tensor.Tensor::stack(batch)
}

pub fn Conv2d::parameters(self : Conv2d) -> Iter[@tensor.Tensor] {
  self.weight.iter().flat_map(fn { channel => channel.iter() })
}
