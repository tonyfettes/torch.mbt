///|
pub(readonly) trait Distribution {
  sample(Self, &@shape.Shape) -> @tensor.Tensor[Float]
  sample_(Self, @tensor.Tensor[Float]) -> Unit
}

///|
pub(readonly) trait Continuous: Distribution {
  rsample(Self, &@shape.Shape) -> @tensor.Tensor[Float]
}

///|
let rand : Ref[@random.Rand] = @ref.new(@random.Rand::new())

///|
pub fn seed(seed : Bytes) -> Unit {
  rand.val = @random.Rand::new(seed~)
}
