///|
pub fn Tensor::squeeze(
  self : Tensor[Float],
  dim? : &@shape.Shape
) -> Tensor[Float] {
  let shape = []
  match dim {
    None =>
      for s in self.shape {
        if s != 1 {
          shape.push(s)
        }
      }
    Some(dim) => {
      let dim = dim.to_fixed_array()
      dim.sort()
      let mut i = 0
      for d in dim {
        while i < d {
          shape.push(self.shape[i])
          i = i + 1
        }
        if self.shape[i] != 1 {
          shape.push(self.shape[i])
        }
        i = i + 1
      }
      while i < self.shape.length() {
        shape.push(self.shape[i])
        i = i + 1
      }
    }
  }
  let shape = FixedArray::from_array(shape)
  Tensor::{
    value: self.value,
    shape,
    refcnt: 0,
    graph: Nop(self),
    grad: self.grad,
  }
}
