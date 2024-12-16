///|
fn power(base : Float, exponent : Int) -> Float {
  if exponent == 0 {
    return 1.0
  }
  if exponent == 1 {
    return base
  }
  let mut result : Float = 1.0
  let mut base = base
  let mut exponent = exponent
  while exponent > 0 {
    if exponent % 2 == 1 {
      result = result * base
    }
    base = base * base
    exponent = exponent / 2
  }
  result
}

test "power" {
  inspect!(power(2.0, 0), content="1")
  inspect!(power(2.0, 1), content="2")
  inspect!(power(0.0, 5), content="0")
  inspect!(power(1.0, 10), content="1")
  inspect!(power(3.0, 3), content="27")
  inspect!(power(4.0, 2), content="16")
  inspect!(power(5.0, 3), content="125")
  inspect!(power(2.0, 10), content="1024")
  inspect!(power(1.5, 3), content="3.375")
  inspect!(power(2.5, 2), content="6.25")
  inspect!(power(2.0, 100), content="1.2676506002282294e+30")
}

///|
pub fn Tensor::pow(self : Tensor[Float], exponent : Int) -> Tensor[Float] {
  let length = self.value.length()
  let output : FixedArray[Float] = FixedArray::make(length, (0.0 : Float))
  for i = 0; i < self.value.length(); i = i + 1 {
    output[i] = power(self.value[i], exponent)
  }
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
    graph: Graph::Pow(self, exponent),
    refcnt: 0,
    grad: FixedArray::make(length, (0.0 : Float)),
  }
}
