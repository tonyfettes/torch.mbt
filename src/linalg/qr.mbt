///|
pub fn qr(
  a : @tensor.Tensor[Float]
) -> (@tensor.Tensor[Float], @tensor.Tensor[Float]) {
  if @debug.DEBUG {
    if a.shape.length() != 2 {
      abort("QR decomposition only supports 2D tensors")
    }
  }
  let m = a.shape[0]
  let n = a.shape[1]
  let u = @unsafe.UnsafeArray::make(n, a)
  let e = @unsafe.UnsafeArray::make(n, a)
  fn proj(
    u : @tensor.Tensor[Float],
    a : @tensor.Tensor[Float]
  ) -> @tensor.Tensor[Float] {
    (u * a).sum() / (u * u).sum() * u
  }

  // let r = @unsafe.UnsafeArray::make(n, a)
  for i in 0..<n {
    let ai = a[:][i].to_tensor()
    let mut proj_sum = @tensor.zeros(m)
    for j in 0..<i {
      proj_sum = proj_sum + proj(u[j], ai)
    }
    let ui = ai - proj_sum
    u[i] = ui
    let ei = ui / vector_norm(ui)
    e[i] = ei
  }
  let qt = @tensor.stack(e._)
  let q = qt.transpose(-1, 0)
  let r = qt.matmul(a).triu()
  (q, r)
}
