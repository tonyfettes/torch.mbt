pub struct Linear {
  weight : @tensor.Tensor
  bias : @tensor.Tensor
} derive(Show, ToJson, FromJson)

pub fn Linear::of(weight : @tensor.Tensor, bias : @tensor.Tensor) -> Linear {
  Linear::{ weight, bias }
}

pub fn Linear::new(
  in_features : Int,
  out_features : Int,
  ~weight_distribution? : @distributions.Continuous,
  ~bias_distribution? : @distributions.Continuous
) -> Linear {
  let in_features_sqrt = in_features.to_double().sqrt()
  let weight_distribution = match weight_distribution {
    None => {
      let in_features_sqrt_inv = 1.0 / in_features_sqrt
      @distributions.Uniform::new(-in_features_sqrt_inv, in_features_sqrt_inv)
      as @distributions.Continuous
    }
    Some(distrib) => distrib
  }
  let bias_distribution = match bias_distribution {
    None => {
      let bound = 1.0 / in_features_sqrt
      @distributions.Uniform::new(-bound, bound) as @distributions.Continuous
    }
    Some(distrib) => distrib
  }
  Linear::{
    weight: weight_distribution.sample([in_features, out_features]),
    bias: bias_distribution.sample([out_features]),
  }
}

pub fn Linear::forward(self : Linear, input : @tensor.Tensor) -> @tensor.Tensor {
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
  let output = input.matmul(self.weight) + self.bias
  let output_shape = input_shape._.copy()
  output_shape[di] = out_features
  output.reshape(output_shape)
}

pub fn Linear::parameters(self : Linear) -> Iter[@tensor.Tensor] {
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
