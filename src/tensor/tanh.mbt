///|
pub fn Tensor::tanh(self : Tensor[Float]) -> Tensor[Float] {
  (self.exp() - (-self).exp()) / (self.exp() + (-self).exp())
}
