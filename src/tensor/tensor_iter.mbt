pub fn Tensor::iter(self : Tensor, dim~ : Int = 0) -> Iter[Tensor] {
  let shape = self.shape
  if shape.length() == 0 {
    abort("Iterating over a scalar tensor is not allowed.")
  }
  let block = self.block
  let shape_length = self.shape.length()
  let dimension = if dim < 0 { shape_length + dim } else { dim }
  if dimension < 0 || dimension >= shape_length {
    abort("Dimension \{dimension} out of range [0, \{shape_length})")
  }
  Iter::new(
    fn(yield_) {
      fn build(d : Int, p : Int) -> IterResult {
        if d == dimension + 1 {
          let shape = FixedArray::makei(
            shape_length - d,
            fn(i) { shape[i + d] },
          )
          let block = compute_block_size(shape)
          let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
          let value = FixedArray::makei(total, fn(i) { self.value[i + p] })
          let index = FixedArray::makei(total, fn(i) { i + p })
          let tensor = Tensor::{
            value,
            shape,
            block,
            graph: Get(self, index),
            refcnt: 0,
            grad: FixedArray::make(total, (0.0 : Float)),
          }
          return yield_(tensor)
        }
        for i in 0..<shape[d] {
          match build(d + 1, p + i * block[d]) {
            IterContinue => ()
            IterEnd => break IterEnd
          }
        } else {
          IterContinue
        }
      }

      ignore(build(0, 0))
      IterEnd
    },
  )
}
