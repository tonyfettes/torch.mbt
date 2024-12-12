///|
pub fn Tensor::flatten(self : Tensor, start~ : Int = 0, end? : Int) -> Tensor {
  let end = match end {
    Some(end) => end
    None => self.shape.length()
  }
  let shape = []
  for i = 0; i < start; i = i + 1 {
    shape.push(self.shape[i])
  }
  if start < end {
    let mut flat_size = 1
    for i = start; i < end; i = i + 1 {
      flat_size *= self.shape[i]
    }
    shape.push(flat_size)
  }
  for i = end; i < self.shape.length(); i = i + 1 {
    shape.push(self.shape[i])
  }
  let shape = FixedArray::from_array(shape)
  let block = compute_block_size(shape)
  Tensor::{
    value: self.value,
    shape,
    block,
    refcnt: 0,
    graph: Nop(self),
    grad: self.grad,
  }
}
