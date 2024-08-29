let rand : Ref[@random.Rand] = @ref.new(@random.Rand::new())

pub fn seed(seed : Bytes) -> Unit {
  rand.val = @random.Rand::new(~seed)
}
