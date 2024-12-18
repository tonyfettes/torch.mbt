///|
pub fn Tensor::sum(
  self : Tensor[Float],
  dim? : &@shape.Shape,
  keep_dim~ : Bool = false
) -> Tensor[Float] {
  let self_shape = self.shape
  let self_block = self.block()
  let dim = match dim {
    None => FixedArray::makei(self_shape.length(), fn(i) { i })
    Some(dim) => dim.to_fixed_array()
  }
  let do_sum = FixedArray::make(self_shape.length(), false)
  for d in dim {
    do_sum[d] = true
  }
  let shape = []
  for i in 0..<self_shape.length() {
    if do_sum[i] {
      if keep_dim {
        shape.push(1)
      }
    } else {
      shape.push(self_shape[i])
    }
  }
  let shape = FixedArray::from_array(shape)
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value : FixedArray[Float] = FixedArray::make(total, 0.0)
  let index : FixedArray[Array[Int]] = FixedArray::makei(total, fn(__) { [] })
  fn build(sum_dim : Int, sum_pos : Int, val_dim : Int, val_pos : Int) -> Unit {
    if val_dim == self_shape.length() {
      value[sum_pos] += self.value[val_pos]
      index[sum_pos].push(val_pos)
      return
    }
    if do_sum[val_dim] {
      if keep_dim {
        for i in 0..<self_shape[val_dim] {
          build(
            sum_dim + 1,
            sum_pos,
            val_dim + 1,
            val_pos + i * self_block[val_dim],
          )
        }
      } else {
        for i in 0..<self_shape[val_dim] {
          build(
            sum_dim,
            sum_pos,
            val_dim + 1,
            val_pos + i * self_block[val_dim],
          )
        }
      }
    } else {
      for i in 0..<self_shape[val_dim] {
        build(
          sum_dim + 1,
          sum_pos + i * block[sum_dim],
          val_dim + 1,
          val_pos + i * self_block[val_dim],
        )
      }
    }
  }

  build(0, 0, 0, 0)
  Tensor::{
    value,
    shape,
    graph: Graph::Sum(self, index),
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
