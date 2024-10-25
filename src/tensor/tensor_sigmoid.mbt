pub fn Tensor::sigmoid(self : Tensor) -> Tensor {
  let ones = ones(self.shape._)
  ones / (ones + exp(-self))
}
