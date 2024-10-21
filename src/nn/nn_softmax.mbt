pub struct Softmax {} derive(Show, ToJson, FromJson)

pub fn Softmax::new() -> Softmax {
  Softmax::{  }
}

pub fn Softmax::output(_self : Softmax, logger : Logger) -> Unit {
  logger.write_string("Softmax()")
}

pub fn Softmax::forward(
  _self : Softmax,
  input : @tensor.Tensor
) -> @tensor.Tensor {
  let exp = input.exp()
  let sum = exp.sum()
  exp / sum
}

pub fn Softmax::parameters(_self : Softmax) -> Iter[@tensor.Tensor] {
  Iter::empty()
}

pub impl @transpile.ToPyTorchSource for Softmax with transpile(
  _self : Softmax,
  transpiler : @transpile.PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.Softmax(dim=-1)")
  variable
}
