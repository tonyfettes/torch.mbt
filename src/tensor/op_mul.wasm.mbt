///|
extern "wasm" fn simd_f32x4_mul(in0 : Int, in1 : Int, out : Int, len : Int) =
  #|(func (param $in0 i32) (param $in1 i32) (param $out i32) (param $len i32)
  #| (local $vec0 v128)
  #| (local $vec1 v128)
  #| (local $prod v128)
  #| (local $end i32)
  #|
  #| (local.set $end (i32.add (local.get $out) (i32.mul (local.get $len) (i32.const 16))))
  #|
  #| (loop $loop
  #|  (local.set $vec0 (v128.load (local.get $in0)))
  #|  (local.set $vec1 (v128.load (local.get $in1)))
  #|  (local.set $prod (f32x4.mul (local.get $vec0) (local.get $vec1)))
  #|  (v128.store (local.get $out) (local.get $prod))
  #|
  #|  (local.set $in0 (i32.add (local.get $in0) (i32.const 16)))
  #|  (local.set $in1 (i32.add (local.get $in1) (i32.const 16)))
  #|  (local.set $out (i32.add (local.get $out) (i32.const 16)))
  #|
  #|  (br_if $loop (i32.lt_u (local.get $out) (local.get $end)))))

///|
fn fixed_array_simd_mul(
  this : @unsafe.UnsafeArray[Float],
  that : @unsafe.UnsafeArray[Float],
  result : @unsafe.UnsafeArray[Float],
  length : Int
) -> Unit {
  let alignment = 4
  let blocks = length / alignment
  if blocks != 0 {
    simd_f32x4_mul(
      this.reinterpret_as_int(),
      that.reinterpret_as_int(),
      result.reinterpret_as_int(),
      length / alignment,
    )
  }
  let residue = length - length % alignment
  for i in residue..<length {
    result[i] = this[i] * that[i]
  }
}

///|
pub fn Tensor::op_mul(
  self : Tensor[Float],
  other : Tensor[Float]
) -> Tensor[Float] {
  let (self, other) = broadcast(self, other)
  let length = self.value.length()
  let output : FixedArray[Float] = FixedArray::make(length, (0.0 : Float))
  fixed_array_simd_mul(self.value, other.value, output, length)
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
    graph: Graph::Mul(self, other),
    refcnt: 0,
    grad: FixedArray::make(length, (0.0 : Float)),
  }
}
