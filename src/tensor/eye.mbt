///|
pub fn eye(n : Int, m? : Int, requires_grad~ : Bool = false) -> Tensor[Float] {
  let m = m.or(n)
  let shape : FixedArray[Int] = [n, m]
  let total = n * m
  let value = FixedArray::make(total, (0.0 : Float))
  for i in 0..<n {
    for j in 0..<m {
      if i == j {
        value[i * m + j] = 1
      } else {
        value[i * m + j] = 0
      }
    }
  }
  Tensor::{
    shape,
    value,
    graph: if requires_grad {
      Graph::Var
    } else {
      Graph::Val
    },
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}
