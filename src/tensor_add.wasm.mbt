extern "wasm" fn simd_f64x2_loop_add(
  in0 : Int,
  in1 : Int,
  out : Int,
  len : Int
) =
  #|(func (param $in0 i32) (param $in1 i32) (param $out i32) (param $len i32)
  #| (local $vec0 v128)
  #| (local $vec1 v128)
  #| (local $sum v128)
  #| (local $end i32)
  #|
  #| (local.set $end (i32.add (local.get $out) (i32.mul (local.get $len) (i32.const 16))))
  #|
  #| (loop $loop
  #|  (local.set $vec0 (v128.load (local.get $in0)))
  #|  (local.set $vec1 (v128.load (local.get $in1)))
  #|  (local.set $sum (f64x2.add (local.get $vec0) (local.get $vec1)))
  #|  (v128.store (local.get $out) (local.get $sum))
  #|
  #|  (local.set $in0 (i32.add (local.get $in0) (i32.const 16)))
  #|  (local.set $in1 (i32.add (local.get $in1) (i32.const 16)))
  #|  (local.set $out (i32.add (local.get $out) (i32.const 16)))
  #|
  #|  (br_if $loop (i32.lt_u (local.get $out) (local.get $end)))))

extern "wasm" fn array_double_address(array : FixedArray[Double]) -> Int =
  #|(func (param i32) (result i32) local.get 0 i32.const 8 i32.add)

fn array_double_simd_add(
  in0 : FixedArray[Double],
  in1 : FixedArray[Double],
  out : FixedArray[Double],
  len : Int
) -> Unit {
  simd_f64x2_loop_add(
    array_double_address(in0),
    array_double_address(in1),
    array_double_address(out),
    len,
  )
}

pub fn Tensor::op_add(self : Tensor, other : Tensor) -> Tensor {
  let length = self.value.length()
  let value : FixedArray[Double] = FixedArray::make(length, 0.0)
  array_double_simd_add(self.value, other.value, value, length)
  Tensor::{
    value,
    shape: self.shape,
    block: self.block,
    graph: Graph::Add(self, other),
    ref: 0,
    grad: FixedArray::make(length, 0.0),
  }
}
