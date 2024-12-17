///|
pub fn Tensor::mean(
  self : Tensor[Float],
  dim? : &@shape.Shape,
  keep_dim~ : Bool = false
) -> Tensor[Float] {
  let len = match dim {
    None => self.value.length()
    Some(dim) => {
      let mut len = 1
      for d in dim {
        len *= self.shape[d]
      }
      len
    }
  }
  self.sum(dim?, keep_dim~) / tensor(len.to_double())
}
