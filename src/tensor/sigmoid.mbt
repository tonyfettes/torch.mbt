///|
pub fn Tensor::sigmoid(self : Tensor[Float]) -> Tensor[Float] {
  let ones = ones(self.shape._)
  ones / (ones + exp(-self))
}
