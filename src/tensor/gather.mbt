///|
pub fn gather(
  input : Tensor[Float],
  dim : Int,
  index : Tensor[Float]
) -> Tensor[Float] {
  let input_shape = input.shape
  let input_block = input.block()
  let index_block = index.block()
  if @debug.DEBUG {
    if input_shape.length() != index.shape.length() {
      abort("`input` and `index` must have the same number of dimensions")
    }
    for d = 0; d < input_shape.length(); d = d + 1 {
      if d != dim {
        if index.shape[d] > input_shape[d] {
          abort("`index.shape[d]` is greater than `input.shape[d]`")
        }
      }
    }
  }
  let total = if input_shape.length() == 0 {
    1
  } else {
    input_shape[0] * input_block[0]
  }
  let out_val = @unsafe.UnsafeArray::make(total, (0.0 : Float))
  let out_idx = @unsafe.UnsafeArray::make(total, 0)
  fn build(d : Int, vp : Int, ip : Int, op : Int) {
    if d == input_shape.length() {
      // We increment `vp` here.
      let vp = vp + index.value[ip].to_double().to_int() * input_block[dim]
      out_val[op] = input.value[vp]
      out_idx[op] = vp
      return
    }
    if d == dim {
      // We don't increment `vp` here along the gather dimension, because the
      // value of `index[..]` is not yet known.
      for i = 0; i < input_shape[d]; i = i + 1 {
        build(d + 1, vp, ip + i * index_block[d], op + i * input_block[d])
      }
      return
    }
    for i = 0; i < index.shape[d]; i = i + 1 {
      build(
        d + 1,
        vp + i * input_block[d],
        ip + i * index_block[d],
        op + i * input_block[d],
      )
    }
  }

  build(0, 0, 0, 0)
  Tensor::{
    value: out_val,
    shape: input.shape,
    graph: Graph::Get(input, out_idx),
    refcnt: 0,
    grad: @unsafe.UnsafeArray::make(total, (0.0 : Float)),
  }
}
