///|
pub(readonly) struct SGD {
  parameters : Array[@tensor.Tensor[Float]]
  learning_rate : Float
}

///|
pub fn SGD::new(
  parameters : Iter[@tensor.Tensor[Float]],
  learning_rate~ : Float
) -> SGD {
  SGD::{ parameters: parameters.to_array(), learning_rate }
}

///|
pub fn SGD::zero_grad(self : SGD) -> Unit {
  for parameter in self.parameters {
    parameter.grad._.fill(0.0)
  }
}

///|
pub fn SGD::step(self : SGD) -> Unit {
  for parameter in self.parameters {
    for i = 0; i < parameter.value.length(); i = i + 1 {
      parameter.value[i] -= parameter.grad[i] * self.learning_rate
    }
  }
}
