///|
pub fn Tensor::iter(self : Tensor[Float]) -> Iter[TensorView[Float]] {
  let self_shape = self.shape
  if self_shape.length() == 0 {
    abort("Iterating over a scalar tensor is not allowed.")
  }
  let shape = FixedArray::from_iter(self_shape.iter().drop(1))
  Iter::new(fn(yield_) {
    for i in 0..<self_shape[0] {
      match yield_(TensorView::{ tensor: self, shape, index: [Index(i)] }) {
        IterContinue => ()
        IterEnd => break IterEnd
      }
    } else {
      IterEnd
    }
  })
}
