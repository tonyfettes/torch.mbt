pub fn Tensor::sigmoid(self : Tensor) -> Tensor {
  let ones = ones(self.shape)
  ones / (ones + exp(-self))
}
