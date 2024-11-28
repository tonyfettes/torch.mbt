struct Uniform {
  lower : @tensor.Tensor
  upper : @tensor.Tensor
}

pub fn Uniform::new[Lower : @tensor.ToTensor, Upper : @tensor.ToTensor](
  lower : Lower,
  upper : Upper
) -> Uniform {
  let (lower, upper) = @tensor.broadcast(lower.to_tensor(), upper.to_tensor())
  Uniform::{ lower, upper }
}

pub fn Uniform::sample(self : Uniform, shape : @shape.Shape) -> @tensor.Tensor {
  let shape = shape.to_fixed_array()
  let sample = @tensor.zeros([..shape, ..self.lower.shape._])
  self.sample_(sample)
  sample
}

pub fn Uniform::sample_(self : Uniform, sample : @tensor.Tensor) -> Unit {
  let lower = self.lower.broadcast_to(sample.shape)
  let upper = self.upper.broadcast_to(sample.shape)
  for i = 0; i < sample.value.length(); i = i + 1 {
    sample.value[i] = rand.val.double().to_float() *
      (upper.value[i] - lower.value[i]) +
      lower.value[i]
  }
}

pub fn Uniform::rsample(self : Uniform, shape : @shape.Shape) -> @tensor.Tensor {
  let shape = shape.to_fixed_array()
  let sample = @tensor.zeros([..shape, ..self.lower.shape._])
  for i = 0; i < sample.value.length(); i = i + 1 {
    sample.value[i] = rand.val.double().to_float()
  }
  sample * (self.upper - self.lower) + self.lower
}

pub fn Uniform::entropy(self : Uniform) -> @tensor.Tensor {
  (self.upper - self.lower).log()
}
