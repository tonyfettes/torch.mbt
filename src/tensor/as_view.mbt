///|
pub fn Tensor::op_as_view[Dtype](
  self : Tensor[Dtype],
  start~ : Int = 0,
  end? : Int
) -> TensorView[Dtype] {
  if self.shape.length() == 0 {
    abort("Tensor is a scalar")
  }
  let end = match end {
    None => self.shape[0]
    Some(end) => {
      if end > self.shape[0] {
        abort(
          "Tensor slice [\{start}, \{end}) out of bounds: [0, \{self.shape[0]})",
        )
      }
      end
    }
  }
  let shape = self.shape._.copy()
  shape[0] = end - start
  TensorView::{ tensor: self, shape, index: [TensorIndex::Slice(start, end)] }
}
