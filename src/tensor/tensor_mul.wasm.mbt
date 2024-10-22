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

fn fixed_array_simd_mul(
  in0 : FixedArray[Float],
  in1 : FixedArray[Float],
  out : FixedArray[Float],
  len : Int
) -> Unit {
  let alignment = 4
  simd_f32x4_mul(
    address_fixed_array_float(in0),
    address_fixed_array_float(in1),
    address_fixed_array_float(out),
    len / alignment,
  )
  let residue = len - len % alignment
  for i in residue..<len {
    out[i] = in0[i] * in1[i]
  }
}

pub fn Tensor::op_mul(self : Tensor, other : Tensor) -> Tensor {
  let length = self.value.length()
  let output : FixedArray[Float] = FixedArray::make(length, 0.0)
  fixed_array_simd_mul(self.value, other.value, output, length)
  Tensor::{
    value: output,
    shape: self.shape,
    block: self.block,
    graph: Graph::Mul(self, other),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}
