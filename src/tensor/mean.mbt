pub fn Tensor::mean(
  self : Tensor,
  dim? : @shape.Shape,
  keep_dim~ : Bool = false
) -> Tensor {
  self.sum() / tensor(self.value.length().to_double())
}
