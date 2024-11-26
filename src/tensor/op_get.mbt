pub fn Tensor::op_get(self : Tensor, index : Int) -> TensorView {
  if @debug.DEBUG {
    if self.shape.length() == 0 {
      abort("Tensor is a scalar")
    }
    if index < 0 || index >= self.shape[0] {
      abort("Tensor element access \{index} out of bounds [0, \{self.shape[0]}")
    }
  }
  TensorView::{ tensor: self, index: [Index(index)] }
}
