///|
pub fn Tensor::max(
  self : Tensor[Float],
  dim? : Int,
  keep_dim~ : Bool = false
) -> (Tensor[Float], Tensor[Int]) {
  let self_block = self.block()
  guard let Some(dim) = dim else {
    None => {
      let mut max_value = self.value[0]
      let mut max_index = 0
      for i in 1..<self.value.length() {
        if self.value[i] > max_value {
          max_value = self.value[i]
          max_index = i
        }
      }
      let value = Tensor::{
        value: [max_value],
        shape: [],
        graph: Graph::Get(self, [max_index]),
        refcnt: 0,
        grad: [0.0],
      }
      let index = Tensor::{
        value: [max_index],
        shape: [],
        graph: Graph::Val,
        refcnt: 0,
        grad: [],
      }
      (value, index)
    }
  }
  if @debug.DEBUG {
    if dim < 0 || dim >= self.shape.length() {
      abort("Invalid dimension \{dim} for max operation.")
    }
  }
  let shape = FixedArray::makei(self.shape.length(), fn(i) {
    if i == dim {
      1
    } else {
      self.shape[i]
    }
  })
  let block = compute_block_size(shape)
  let total = shape[0] * block[0]
  let value = FixedArray::make(total, @float.min_value)
  let value_index = FixedArray::make(total, -1)
  let index = FixedArray::make(total, 0)
  fn build(d : Int, max_pos : Int, val_pos : Int, i : Int) {
    if d == self.shape.length() {
      if self.value[val_pos] > value[max_pos] {
        value[max_pos] = self.value[val_pos]
        value_index[max_pos] = val_pos
        index[max_pos] = i
      }
      return
    }
    if d == dim {
      for i in 0..<self.shape[d] {
        build(d + 1, max_pos, val_pos + i * self_block[d], i)
      }
      return
    }
    for j in 0..<self.shape[d] {
      build(d + 1, max_pos + j * block[d], val_pos + j * self_block[d], i)
    }
  }

  build(0, 0, 0, 0)
  let shape = if keep_dim {
    shape
  } else {
    FixedArray::makei(self.shape.length() - 1, fn(i) {
      if i < dim {
        self.shape[i]
      } else {
        self.shape[i + 1]
      }
    })
  }
  let value = Tensor::{
    value,
    shape,
    graph: Get(self, value_index),
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
  let index = Tensor::{ value: index, shape, graph: Val, refcnt: 0, grad: [] }
  (value, index)
}
