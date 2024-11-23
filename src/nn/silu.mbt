struct SiLU {} derive(Show, ToJson, FromJson)

pub fn SiLU::new() -> SiLU {
  SiLU::{  }
}

pub fn SiLU::output(_self : SiLU, logger : Logger) -> Unit {
  logger.write_string("SiLU()")
}

pub fn SiLU::forward(_self : SiLU, input : @tensor.Tensor) -> @tensor.Tensor {
  @functional.silu(input)
}

pub fn SiLU::parameters(_self : SiLU) -> Iter[@tensor.Tensor] {
  Iter::empty()
}
