pub fn Tensor::sum(self : Tensor, dim~ : @shape.Shape = ()) -> Tensor {
  let dim = dim.to_fixed_array()
  let dim = if dim.length() == 0 {
    FixedArray::makei(self.shape.length(), fn(i) { i })
  } else {
    dim
  }
  let do_sum = FixedArray::make(self.shape.length(), false)
  for d in dim {
    do_sum[d] = true
  }
  let shape = []
  for d in 0..<self.shape.length() {
    if not(do_sum[d]) {
      shape.push(self.shape[d])
    }
  }
  let shape = FixedArray::from_array(shape)
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value : FixedArray[Float] = FixedArray::make(total, 0.0)
  let index : FixedArray[Array[Int]] = FixedArray::makei(total, fn(__) { [] })
  fn build(d : Int, sum_pos : Int, val_pos : Int) -> Unit {
    if d == self.shape.length() {
      value[sum_pos] += self.value[val_pos]
      index[sum_pos].push(val_pos)
      return
    }
    if do_sum[d] {
      for i in 0..<self.shape[d] {
        build(d + 1, sum_pos, val_pos + i * self.block[d])
      }
    } else {
      for i in 0..<self.shape[d] {
        build(d + 1, sum_pos + i * self.block[d], val_pos + i * self.block[d])
      }
    }
  }

  build(0, 0, 0)
  Tensor::{
    value,
    shape,
    block,
    graph: Graph::Sum(self, index),
    ref: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
