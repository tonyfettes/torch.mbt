pub fn Tensor::op_as_view(
  self : Tensor,
  ~start : Int,
  ~end? : Int
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
  let index = [TensorIndex::Slice(start, end)]
  TensorView::{ tensor: self, index }
}
