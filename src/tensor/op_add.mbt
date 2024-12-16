///|
trait TensorAdd {
  op_add(Tensor[Self], Tensor[Self]) -> Tensor[Self]
}

///|
type GraphAdd (Tensor[Float], Tensor[Float]) derive(Show)

///|
impl Backward for GraphAdd with incref(self : GraphAdd) -> Unit {
  incref(self._.0)
  incref(self._.1)
}

///|
impl Backward for GraphAdd with propagate(
  self : GraphAdd,
  grad : @unsafe.UnsafeArray[Float]
) -> Unit {
  let (a, b) = self._
  let a_grad = a.grad
  let b_grad = b.grad
  for i in 0..<grad.length() {
    a_grad[i] += grad[i]
    b_grad[i] += grad[i]
  }
  a.propagate()
  b.propagate()
}

///|
impl Backward for GraphAdd with check(self : GraphAdd) -> Unit {
  self._.0.check()
  self._.1.check()
}

///|
pub impl TensorAdd for Int with op_add(self : Tensor[Int], other : Tensor[Int]) -> Tensor[
  Int,
] {
  let length = self.value.length()
  let value = FixedArray::make(length, 0)
  for i = 0; i < length; i = i + 1 {
    value[i] = self.value[i] + other.value[i]
  }
  Tensor::{
    value,
    shape: self.shape,
    block: self.block,
    graph: Val,
    refcnt: 0,
    grad: FixedArray::make(length, (0.0 : Float)),
  }
}

///|
pub fn Tensor::op_add[Dtype : TensorAdd](
  self : Tensor[Dtype],
  other : Tensor[Dtype]
) -> Tensor[Dtype] {
  let (self, other) = Tensor::broadcast(self, other)
  TensorAdd::op_add(self, other)
}
