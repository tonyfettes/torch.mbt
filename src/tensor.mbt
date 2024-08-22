pub type Tensor Array[Value]

pub fn Tensor::val(value : Array[Double]) -> Tensor {
  value.map(Value::val)
}

pub fn Tensor::var(value : Array[Double]) -> Tensor {
  value.map(Value::var)
}

pub fn Tensor::op_get(self : Tensor, index : Int) -> Value {
  self.0[index]
}

pub fn Tensor::op_set(self : Tensor, index : Int, value : Value) -> Unit {
  self.0[index] = value
}

pub fn Tensor::map(self : Tensor, f : (Value) -> Value) -> Tensor {
  self.0.map(f)
}

pub fn Tensor::iter(self : Tensor) -> Iter[Value] {
  self.0.iter()
}

pub fn Tensor::output(self : Tensor, logger : Logger) -> Unit {
  Show::output(self.0, logger)
}

pub fn Tensor::length(self : Tensor) -> Int {
  self.0.length()
}

pub fn Tensor::sample(self : Tensor) -> Int {
  let mut sum = 0.0
  for value in self.0 {
    sum = sum + value.value
  }
  let mut choice = @random.gen_double(random_state) / 2.0 * sum
  for index, value in self.0 {
    choice = choice - value.value
    if choice <= 0.0 {
      return index
    }
  }
  @random.gen_int(random_state) / self.0.length()
}
