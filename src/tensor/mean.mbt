pub fn Tensor::mean(
  self : Tensor,
  dim? : @shape.Shape,
  keep_dim~ : Bool = false
) -> Tensor {
  self.sum(dim?, keep_dim~) / tensor(self.value.length().to_double())
}
