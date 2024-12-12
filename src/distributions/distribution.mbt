///|
pub(readonly) trait Distribution {
  sample(Self, @shape.Shape) -> @tensor.Tensor
  sample_(Self, @tensor.Tensor) -> Unit
}

///|
pub(readonly) trait Continuous: Distribution {
  rsample(Self, @shape.Shape) -> @tensor.Tensor
}

///|
let rand : Ref[@random.Rand] = @ref.new(@random.Rand::new())

///|
pub fn seed(seed : Bytes) -> Unit {
  rand.val = @random.Rand::new(seed~)
}
