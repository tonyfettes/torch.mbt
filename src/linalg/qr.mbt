enum QRMode {
  Reduced
  Complete
}

pub fn qr(
  input : @tensor.Tensor,
  mode~ : QRMode = Reduced
) -> (@tensor.Tensor, @tensor.Tensor) {
  if @debug.DEBUG {
    if input.shape.length() != 2 {
      abort("QR decomposition only supports 2D tensors")
    }
  }
  let m = input.shape[0]
  let n = input.shape[1]
  for i in 0..<n {
    let column = input[:][i]
    let all_zero = true
    for value in column.iter() {
      if value.to_double() != 0.0 {
        all_zero = false
        break
      }
    }
  }
  ((), ())
}
