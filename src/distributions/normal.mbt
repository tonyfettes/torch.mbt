///|
let normal_buffer : Array[Float] = []

///|
fn normal(mean~ : Float = 0.0, std~ : Float = 1.0) -> Float {
  match normal_buffer.pop() {
    None => ()
    Some(z1) => return z1
  }
  let u1 = rand.val.double()
  let u2 = rand.val.double()
  let r = (-2.0 * u1.ln()).sqrt()
  let z0 = r * @math.cos(2.0 * @math.PI * u2)
  let z1 = r * @math.sin(2.0 * @math.PI * u2)
  normal_buffer.push(z1.to_float())
  return z0.to_float() * std + mean
}

///|
struct Normal {
  loc : @tensor.Tensor[Float]
  scale : @tensor.Tensor[Float]
}

///|
pub fn Normal::new[Loc : @tensor.ToValue, Scale : @tensor.ToValue](
  loc : Loc,
  scale : Scale
) -> Normal {
  let (loc, scale) = @tensor.broadcast(
    @tensor.tensor(loc),
    @tensor.tensor(scale),
  )
  Normal::{ loc, scale }
}

///|
pub fn Normal::sample(
  self : Normal,
  shape : &@shape.Shape
) -> @tensor.Tensor[Float] {
  let shape = shape.to_fixed_array()
  let sample = @tensor.zeros([..shape, ..self.loc.shape._])
  self.sample_(sample)
  sample
}

///|
pub fn Normal::sample_(self : Normal, sample : @tensor.Tensor[Float]) -> Unit {
  let loc = self.loc.broadcast_to(sample.shape)
  let scale = self.scale.broadcast_to(sample.shape)
  for i = 0; i < sample.value.length(); i = i + 1 {
    sample.value[i] = normal(mean=loc.value[i], std=scale.value[i])
  }
}

///|
pub fn Normal::rsample(
  self : Normal,
  shape : &@shape.Shape
) -> @tensor.Tensor[Float] {
  let shape = shape.to_fixed_array()
  let sample = @tensor.zeros([..shape, ..self.loc.shape._])
  for i = 0; i < sample.value.length(); i = i + 1 {
    sample.value[i] = normal(mean=0, std=1)
  }
  sample * self.scale + self.loc
}

///|
pub let standard_normal : Normal = Normal::new(0.0, 1.0)

///|
pub fn Normal::entropy(self : Normal) -> @tensor.Tensor[Float] {
  @tensor.tensor((2.0 * @math.PI).ln() * 0.5 + 0.5) + self.scale.log()
}
