pub fn Tensor::op_get(self : Tensor, index : Int) -> TensorView {
  @debug.assert(self.shape.length() > 0, "Tensor is a scalar")
  @debug.assert(
    index >= 0 && index < self.shape[0],
    "Tensor element access \{index} out of bounds [0, \{self.shape[0]}",
  )
  TensorView::{ tensor: self, index: [Index(index)] }
}
