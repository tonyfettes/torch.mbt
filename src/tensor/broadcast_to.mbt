///|
pub fn Tensor::broadcast_to[Shape : @shape.Shape](
  self : Tensor[Float],
  shape : Shape
) -> Tensor[Float] {
  let shape = shape.to_fixed_array()
  let self_dim = self.shape.length()
  let dimension = shape.length()
  if @debug.DEBUG && dimension < self_dim {
    abort("Cannot broadcast to a shape of lower dimension")
  }
  let self_shape = FixedArray::make(dimension, 1)
  let self_offset = dimension - self_dim
  for i = 0; i < self_dim; i = i + 1 {
    self_shape[i + self_offset] = self.shape[i]
  }
  let self_block = compute_block_size(self_shape)
  let block = compute_block_size(shape)
  let total = if dimension == 0 { 1 } else { shape[0] * block[0] }
  let value = Array::new(capacity=total)
  let index = Array::new(capacity=total)
  fn build(d : Int, p : Int) -> Unit {
    if d == dimension {
      value.push(self.value[p])
      index.push(p)
      return
    }
    if self_shape[d] == 1 {
      for i = 0; i < shape[d]; i = i + 1 {
        build(d + 1, p)
      }
      return
    }
    if self_shape[d] != shape[d] {
      abort("Shapes \{self_shape} and \{shape} are not broadcast-able")
    }
    for i = 0; i < self_shape[d]; i = i + 1 {
      build(d + 1, p + i * self_block[d])
    }
  }

  build(0, 0)
  Tensor::{
    value: FixedArray::from_array(value),
    shape,
    block,
    graph: Graph::Get(self, FixedArray::from_array(index)),
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
