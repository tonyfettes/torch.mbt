struct Sigmoid {} derive(Show, ToJson, FromJson)

pub fn Sigmoid::new() -> Sigmoid {
  Sigmoid::{  }
}

pub fn Sigmoid::forward(
  _self : Sigmoid,
  input : @tensor.Tensor
) -> @tensor.Tensor {
  input.sigmoid()
}

pub fn Sigmoid::parameters(_self : Sigmoid) -> Iter[@tensor.Tensor] {
  return Iter::empty()
}

pub impl @transpile.ToPyTorchSource for Sigmoid with transpile(
  _self : Sigmoid,
  transpiler : @transpile.PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.Sigmoid()")
  variable
}
