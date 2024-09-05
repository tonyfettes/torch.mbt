pub struct SGD {
  parameters : Array[Tensor]
  learning_rate : Double
}

pub fn SGD::new(parameters : Iter[Tensor], ~learning_rate : Double) -> SGD {
  let vars = []
  for parameter in parameters {
    parameter.graph = Var
    vars.push(parameter)
  }
  SGD::{ parameters: parameters.to_array(), learning_rate }
}

pub fn SGD::zero_grad(self : SGD) -> Unit {
  for parameter in self.parameters {
    parameter.grad.fill(0.0)
  }
}

pub fn SGD::step(self : SGD) -> Unit {
  for parameter in self.parameters {
    for i = 0; i < parameter.value.length(); i = i + 1 {
      parameter.value[i] -= parameter.grad[i] * self.learning_rate
    }
  }
}
