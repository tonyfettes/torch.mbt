pub fn Tensor::max(self : Tensor) -> Tensor {
  if self.value.length() == 0 {
    abort("Cannot take the maximum of an empty tensor")
  }
  let mut max_value = self.value[0]
  let mut max_index = 0
  for i in 1..<self.value.length() {
    if self.value[i] > max_value {
      max_value = self.value[i]
      max_index = i
    }
  }
  Tensor::{
    value: [max_value],
    shape: [],
    block: [],
    graph: Graph::Get(self, [max_index]),
    refcnt: 0,
    grad: [0.0],
  }
}
