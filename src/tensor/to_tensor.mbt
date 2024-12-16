///|
enum Value {
  Bool(Tensor[Bool])
  Int(Tensor[Int])
  Float(Tensor[Float])
  Double(Tensor[Double])
  Array(FixedArray[Value])
}

///|
trait ToValue {
  to_value(Self) -> Value
}

///|
trait ValueToTensor {
  value_to_tensor(Value) -> Tensor[Self]
}

///|
fn Tensor::scalar[T](value : T) -> Tensor[T] {
  Tensor::{
    value: [value],
    shape: [],
    block: [],
    graph: Val,
    refcnt: 0,
    grad: [0.0],
  }
}

///|
pub impl ToValue for Bool with to_value(self : Bool) -> Value {
  Value::Bool(scalar(self))
}

///|
pub impl Default for Bool with default() -> Bool { false }

///|
pub impl ToValue for Int with to_value(self : Int) -> Value {
  Value::Int(scalar(self))
}

///|
pub impl Default for Int with default() -> Int { 0 }

///|
pub impl ToValue for Float with to_value(self : Float) -> Value {
  Value::Float(scalar(self))
}

///|
pub impl Default for Float with default() -> Float { 0.0 }

///|
pub impl ToValue for Double with to_value(self : Double) -> Value {
  Value::Double(scalar(self))
}

///|
pub impl Default for Double with default() -> Double { 0.0 }

///|
pub impl[T : ToValue] ToValue for FixedArray[T] with to_value(
  self : FixedArray[T]
) -> Value {
  Value::Array(self.map(fn { x => x.to_value() }))
}

///|
pub impl[X] Default for FixedArray[X] with default() -> FixedArray[X] { [] }

///|
pub impl[T : ToValue] ToValue for Array[T] with to_value(self : Array[T]) -> Value {
  Value::Array(FixedArray::from_iter(self.iter().map(fn { x => x.to_value() })))
}

///|
pub impl[X] Default for Array[X] with default() -> Array[X] { [] }

///|
trait TensorToValue {
  tensor_to_value(Tensor[Self]) -> Value
}

///|
pub impl TensorToValue for Float with tensor_to_value(self : Tensor[Float]) -> Value {
  Value::Float(self)
}

///|
pub impl[T : TensorToValue] ToValue for Tensor[T] with to_value(
  self : Tensor[T]
) -> Value {
  T::tensor_to_value(self)
}

///|
trait ToFloat {
  to_float(Self) -> Float
}

///|
impl ToFloat for Bool with to_float(self : Bool) -> Float {
  self.to_int().to_float()
}

///|
pub fn Tensor::to_float[Dtype : ToFloat](self : Tensor[Dtype]) -> Tensor[Float] {
  let value = @unsafe.UnsafeArray::new(self.value.length())
  for i in 0..<self.value.length() {
    value[i] = self.value[i].to_float()
  }
  Tensor::{
    shape: self.shape,
    block: self.block,
    value,
    graph: Val,
    refcnt: 0,
    grad: @unsafe.UnsafeArray::make(self.value.length(), (0.0 : Float)),
  }
}

///|
trait ToInt {
  to_int(Self) -> Int
}

///|
pub fn Tensor::to_int[Dtype : ToInt](self : Tensor[Dtype]) -> Tensor[Int] {
  let value = @unsafe.UnsafeArray::new(self.value.length())
  for i in 0..<self.value.length() {
    value[i] = self.value[i].to_int()
  }
  Tensor::{
    shape: self.shape,
    block: self.block,
    value,
    graph: Val,
    refcnt: 0,
    grad: @unsafe.UnsafeArray::make(self.value.length(), (0.0 : Float)),
  }
}

///|
pub impl ValueToTensor for Float with value_to_tensor(value : Value) -> Tensor[
  Float,
] {
  fn build(dim : Int, val : Value) -> Tensor[Float] {
    match val {
      Value::Array(array) => {
        let tensors = array.map(fn { value => build(dim + 1, value) })
        Tensor::stack(tensors)
      }
      Value::Bool(bool) => bool.to_float()
      Value::Int(int) => int.to_float()
      Value::Float(float) => float
      Value::Double(double) => double.to_float()
    }
  }

  build(0, value)
}

///|
pub impl ValueToTensor for Int with value_to_tensor(value : Value) -> Tensor[
  Int,
] {
  fn build(dim : Int, val : Value) -> Tensor[Int] {
    match val {
      Value::Array(array) => {
        let tensors = array.map(fn { value => build(dim + 1, value) })
        Tensor::stack(tensors)
      }
      Value::Bool(bool) => bool.to_int()
      Value::Int(int) => int
      Value::Float(float) => float.to_int()
      Value::Double(double) => double.to_int()
    }
  }

  build(0, value)
}
