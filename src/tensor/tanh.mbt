pub fn Tensor::tanh(self : Tensor) -> Tensor {
  (self.exp() - (-self).exp()) / (self.exp() + (-self).exp())
}
