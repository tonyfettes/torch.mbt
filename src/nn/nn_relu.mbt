pub struct ReLU {} derive(Show, ToJson, FromJson)

pub fn ReLU::new() -> ReLU {
  ReLU::{  }
}

pub fn ReLU::output(_self : ReLU, logger : Logger) -> Unit {
  logger.write_string("ReLU()")
}

pub fn ReLU::forward(_self : ReLU, input : @tensor.Tensor) -> @tensor.Tensor {
  input.reLU()
}

pub fn ReLU::parameters(_self : ReLU) -> Iter[@tensor.Tensor] {
  Iter::empty()
}

pub impl @transpile.ToPyTorchSource for ReLU with transpile(
  _self : ReLU,
  transpiler : @transpile.PyTorchTranspiler
) -> String {
  let variable = transpiler.allocate()
  transpiler.push_model("\{variable} = torch.nn.ReLU()")
  variable
}

pub struct ReLU6 {}
