pub struct Linear {
  weight : Tensor
  bias : Tensor
} derive(Show, ToJson, FromJson)

pub fn Linear::of(weight : Tensor, bias : Tensor) -> Linear {
  Linear::{ weight, bias }
}

pub fn Linear::new(
  in_features : Int,
  out_features : Int,
  ~weight_distribution? : Continuous,
  ~bias_distribution? : Continuous
) -> Linear {
  let in_features_sqrt = in_features.to_double().sqrt()
  let weight_distribution = match weight_distribution {
    None => {
      let in_features_sqrt_inv = 1.0 / in_features_sqrt
      Uniform::new(-in_features_sqrt_inv, in_features_sqrt_inv) as Continuous
    }
    Some(distrib) => distrib
  }
  let bias_distribution = match bias_distribution {
    None => {
      let bound = 1.0 / in_features_sqrt
      Uniform::new(-bound, bound) as Continuous
    }
    Some(distrib) => distrib
  }
  Linear::{
    weight: weight_distribution.sample([in_features, out_features]),
    bias: bias_distribution.sample([out_features]),
  }
}

pub fn Linear::forward(self : Linear, input : Tensor) -> Tensor {
  let input_shape = input.shape
  let dimension = input_shape.length()
  if dimension == 0 {
    abort("Input tensor should have at least one dimension")
  }
  let mut di = 0
  let mut length = 1
  while di < dimension - 1 {
    length *= input_shape[di]
    di += 1
  }
  let in_features = self.weight.shape[0]
  let out_features = self.weight.shape[1]
  if input_shape[di] != in_features {
    abort(
      "The shape of last dimension of input tensor \{input_shape[di]} does not match the number of input features \{in_features}",
    )
  }
  let input = input.reshape([length, input_shape[di]])
  let output = (input.matmul(self.weight) + self.bias).to_tensor()
  let output_shape = input_shape.copy()
  output_shape[di] = out_features
  output.reshape(output_shape)
}

pub fn Linear::parameters(self : Linear) -> Iter[Tensor] {
  [self.weight, self.bias].iter()
}

pub fn Linear::output(self : Linear, logger : Logger) -> Unit {
  logger.write_string("Linear(\n")
  for row = 0; row < self.weight.length(); row = row + 1 {
    logger.write_string("  ")
    Show::output(self.weight[row], logger)
    logger.write_string("\t| ")
    Show::output(self.bias[row], logger)
    logger.write_char('\n')
  }
  logger.write_string(")")
}

pub impl ToPyTorchSource for Linear with transpile(
  self : Linear,
  transpiler : PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  let weight_shape = self.weight.shape.copy()
  let dimension = weight_shape.length()
  let width = weight_shape[dimension - 1]
  let height = weight_shape[dimension - 2]
  weight_shape[dimension - 1] = height
  weight_shape[dimension - 2] = width
  let weight_shape = Array::makei(dimension, fn(i) { weight_shape[i].to_string() })
  let weight_shape = String::concat(weight_shape, separator=", ")
  transpiler.push_model(
    "\{variable} = torch.nn.Linear(\{self.weight.shape[0]}, \{self.weight.shape[1]})",
  )
  transpiler.push_after(
    "\{variable}.weight = torch.nn.Parameter(torch.tensor(\{self.weight.to_json().stringify()}).reshape(\{weight_shape}))",
  )
  transpiler.push_after(
    "\{variable}.bias = torch.nn.Parameter(torch.tensor(\{self.bias.to_json().stringify()}))",
  )
  variable
}
