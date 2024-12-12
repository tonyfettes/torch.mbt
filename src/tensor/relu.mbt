///| @alert unstable "Non-PyTorch API"
pub fn Tensor::relu(self : Tensor) -> Tensor {
  Tensor::{
    value: self.value._.map(fn { x => if x > 0.0 { x } else { (0.0 : Float) } }),
    shape: self.shape,
    block: self.block,
    graph: Graph::ReLU(self),
    refcnt: 0,
    grad: FixedArray::make(self.value.length(), (0.0 : Float)),
  }
}
