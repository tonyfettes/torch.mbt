///|
pub fn Tensor::broadcast(self : Tensor, other : Tensor) -> (Tensor, Tensor) {
  let self_dim = self.shape.length()
  let other_dim = other.shape.length()
  let dim = @math.maximum(self_dim, other_dim)
  let self_shape = FixedArray::make(dim, 1)
  let other_shape = FixedArray::make(dim, 1)
  let broad_shape = FixedArray::make(dim, 1)
  let this_offset = dim - self_dim
  for i = 0; i < self_dim; i = i + 1 {
    self_shape[i + this_offset] = self.shape[i]
  }
  let that_offset = dim - other_dim
  for i = 0; i < other_dim; i = i + 1 {
    other_shape[i + that_offset] = other.shape[i]
  }
  for i = 0; i < dim; i = i + 1 {
    broad_shape[i] = @math.maximum(self_shape[i], other_shape[i])
  }
  let self_block = compute_block_size(self_shape)
  let other_block = compute_block_size(other_shape)
  let broad_block = compute_block_size(broad_shape)
  let total = if dim == 0 { 1 } else { broad_shape[0] * broad_block[0] }
  let self_value = Array::new(capacity=total)
  let self_index = Array::new(capacity=total)
  let other_value = Array::new(capacity=total)
  let other_index = Array::new(capacity=total)
  fn build_self(d : Int, p : Int) -> Unit {
    if d == dim {
      self_value.push(self.value[p])
      self_index.push(p)
      return
    }
    if self_shape[d] == 1 {
      for i = 0; i < other_shape[d]; i = i + 1 {
        build_self(d + 1, p)
      }
      return
    }
    if other_shape[d] != 1 && self_shape[d] != other_shape[d] {
      abort("Shapes \{self_shape} and \{other_shape} are not broadcast-able")
    }
    for i = 0; i < self_shape[d]; i = i + 1 {
      build_self(d + 1, p + i * self_block[d])
    }
  }

  fn build_other(d : Int, p : Int) -> Unit {
    if d == dim {
      other_value.push(other.value[p])
      other_index.push(p)
      return
    }
    if other_shape[d] == 1 {
      for i = 0; i < self_shape[d]; i = i + 1 {
        build_other(d + 1, p)
      }
      return
    }
    if self_shape[d] != 1 && self_shape[d] != other_shape[d] {
      abort("Shapes \{self_shape} and \{other_shape} are not broadcast-able")
    }
    for i = 0; i < other_shape[d]; i = i + 1 {
      build_other(d + 1, p + i * other_block[d])
    }
  }

  build_self(0, 0)
  build_other(0, 0)
  (
    Tensor::{
      value: FixedArray::from_array(self_value),
      shape: broad_shape,
      block: broad_block,
      graph: Graph::Get(self, FixedArray::from_array(self_index)),
      refcnt: 0,
      grad: FixedArray::make(total, (0.0 : Float)),
    },
    Tensor::{
      value: FixedArray::from_array(other_value),
      shape: broad_shape,
      block: broad_block,
      graph: Graph::Get(other, FixedArray::from_array(other_index)),
      refcnt: 0,
      grad: FixedArray::make(total, (0.0 : Float)),
    },
  )
}
