///|
type GraphAdd (Tensor, Tensor) derive(Show)

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
