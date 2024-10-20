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
  let bias = Array::makei(
    out_channels,
    fn(__) { weight_distribution.sample([]) },
  )
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
        correlations[ic] = correlate2d(input, weight, padding=self.padding.0)
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

pub impl ToPyTorchSource for Conv2d with transpile(
  self : Conv2d,
  transpiler : PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  let kernel_size = self.weight[0][0].shape
  let kernel_size = (kernel_size[0], kernel_size[1])
  transpiler.push_model(
    "\{variable} = torch.nn.Conv2d(\{self.weight.length()}, \{self.weight[0].length()}, kernel_size=\{kernel_size}, padding=\{self.padding})",
  )
  transpiler.push_after(
    "\{variable}.weight = torch.nn.Parameter(torch.tensor(\{self.weight.to_json().stringify()}))",
  )
  transpiler.push_after(
    "\{variable}.bias = torch.nn.Parameter(torch.tensor(\{self.bias.to_json().stringify()}))",
  )
  variable
}
