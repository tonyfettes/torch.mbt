///|
pub fn Tensor::op_as_view(
  self : Tensor[Float],
  start~ : Int,
  end? : Int
) -> TensorView {
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
