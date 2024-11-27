struct RMSNorm {
  normalized_shape : FixedArray[Int]
  weight : @tensor.Tensor
  eps : Double?
} derive(Show, ToJson, FromJson)

pub fn RMSNorm::new[Shape : @shape.Shape](
  normalized_shape : Shape,
  eps? : Float
) -> RMSNorm {
  let normalized_shape = normalized_shape.to_fixed_array()
  let weight = @tensor.ones(normalized_shape)
  let eps : Double? = match eps {
    None => None
    Some(eps) => Some(eps.to_double())
  }
  RMSNorm::{ normalized_shape, weight, eps }
}

pub fn RMSNorm::forward(
  self : RMSNorm,
  input : @tensor.Tensor
) -> @tensor.Tensor {
  match self.eps {
    Some(eps) =>
      @functional.rms_norm(
        input,
        self.normalized_shape,
        self.weight,
        eps=eps.to_float(),
      )
    None => @functional.rms_norm(input, self.normalized_shape, self.weight)
  }
}

pub fn RMSNorm::parameters(self : RMSNorm) -> Iter[@tensor.Tensor] {
  Iter::singleton(self.weight)
}
