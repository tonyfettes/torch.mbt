/// @alert unstable "Non-PyTorch API"
pub fn Tensor::silu(self : Tensor) -> Tensor {
  self * self.sigmoid()
}
