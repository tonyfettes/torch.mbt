pub trait Tensor {
  tensor(Self) -> @tensor.Tensor
}

pub fn Tensor::new[X : Tensor](self : X) -> Tensor {
  self.tensor()
}
