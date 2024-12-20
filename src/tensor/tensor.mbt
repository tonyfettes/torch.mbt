///|
pub(readonly) struct Tensor[Dtype] {
  value : @unsafe.UnsafeArray[Dtype]
  shape : @unsafe.UnsafeArray[Int]
  block : @unsafe.UnsafeArray[Int]
  priv mut graph : Graph[Dtype]
  priv mut refcnt : Int
  grad : @unsafe.UnsafeArray[Float]
}

///|
pub fn Tensor::requires_grad(self : Tensor[Float]) -> Bool {
  match self.graph {
    Var => true
    _ => false
  }
}

///|
pub fn Tensor::requires_grad_[Dtype](
  self : Tensor[Dtype],
  requires_grad : Bool
) -> Unit {
  if requires_grad {
    self.graph = Var
  } else {
    self.graph = Val
  }
}

///|
enum Graph[Dtype] {
  Val
  Var
  Add(Tensor[Dtype], Tensor[Dtype])
  Sub(Tensor[Dtype], Tensor[Dtype])
  Mul(Tensor[Dtype], Tensor[Dtype])
  Div(Tensor[Dtype], Tensor[Dtype])
  Neg(Tensor[Dtype])
  Exp(Tensor[Dtype])
  Log(Tensor[Dtype])
  Abs(Tensor[Dtype])
  Sqrt(Tensor[Dtype])
  Sum(Tensor[Dtype], @unsafe.UnsafeArray[Array[Int]])
  Pow(Tensor[Dtype], Int)
  MatMul(Tensor[Dtype], Tensor[Dtype])
  Get(Tensor[Dtype], @unsafe.UnsafeArray[Int])
  Cat(@unsafe.UnsafeArray[Tensor[Dtype]], @unsafe.UnsafeArray[(Int, Int)])
  Nop(Tensor[Dtype])
  ReLU(Tensor[Dtype])
  AvgPool2d(Tensor[Dtype], kernel_size~ : (Int, Int), stride~ : (Int, Int))
  MaxPool2d(Tensor[Dtype], @unsafe.UnsafeArray[Int])
  Correlate2d(
    image~ : Tensor[Dtype],
    kernel~ : Tensor[Dtype],
    stride~ : (Int, Int),
    padding~ : (Int, Int)
  )
} derive(Show)

///|
pub fn Tensor::output[Float : Show](
  self : Tensor[Float],
  logger : &Logger
) -> Unit {
  let shape = self.shape
  let dimension = shape.length()
  let value = self.value
  fn build(d : Int, p : Int) {
    if d == dimension {
      Show::output(value[p], logger)
      return
    }
    logger.write_char('[')
    for i = 0; i < shape[d]; i = i + 1 {
      build(d + 1, p + i * self.block[d])
      if i < shape[d] - 1 {
        logger.write_string(", ")
      }
    }
    logger.write_char(']')
  }

  build(0, 0)
}

///|
pub fn Tensor::to_string[Float : Show](self : Tensor[Float]) -> String {
  Show::to_string(self.value)
}

///|
pub fn Tensor::length(self : Tensor[Float]) -> Int {
  self.shape[0]
}

///|
pub(readonly) trait Default {
  default() -> Self
}

///|
fn compute_block_size(shape : FixedArray[Int]) -> FixedArray[Int] {
  let mut dimension = shape.length()
  let block = FixedArray::make(dimension, 1)
  dimension -= 1
  while dimension > 0 {
    block[dimension - 1] = block[dimension] * shape[dimension]
    dimension -= 1
  }
  block
}

test "compute_block_size" {
  inspect!(compute_block_size([2, 3, 4]), content="[12, 4, 1]")
}

///|
fn Tensor::incref[Float](self : Tensor[Float]) -> Unit {
  if self.refcnt == -1 {
    self.refcnt = 0
  }
  self.refcnt += 1
  if self.refcnt > 1 {
    return
  }
  match self.graph {
    Val => ()
    Var => ()
    Add(a, b) => {
      a.incref()
      b.incref()
    }
    Sub(a, b) => {
      a.incref()
      b.incref()
    }
    Mul(a, b) => {
      a.incref()
      b.incref()
    }
    Div(a, b) => {
      a.incref()
      b.incref()
    }
    Neg(x) => x.incref()
    Exp(x) => x.incref()
    Log(x) => x.incref()
    Abs(x) => x.incref()
    Sqrt(x) => x.incref()
    Sum(x, _) => x.incref()
    Pow(x, _) => x.incref()
    MatMul(a, b) => {
      a.incref()
      b.incref()
    }
    Cat(xs, _) =>
      for i in 0..<xs.length() {
        xs[i].incref()
      }
    Get(x, _) => x.incref()
    Nop(x) => x.incref()
    ReLU(x) => x.incref()
    AvgPool2d(x, ..) => x.incref()
    MaxPool2d(x, _) => x.incref()
    Correlate2d(image~, kernel~, ..) => {
      image.incref()
      kernel.incref()
    }
  }
}

///|
fn Tensor::propagate(self : Tensor[Float]) -> Unit {
  if self.refcnt == 0 {
    abort("Propagate through a dead tensor: \{self}")
  }
  self.refcnt -= 1
  if self.refcnt != 0 {
    return
  }
  let self_grad = self.grad
  match self.graph {
    Val => return
    Var => return
    Add(a, b) => {
      let a_grad = a.grad
      let b_grad = b.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i]
        b_grad[i] += self_grad[i]
      }
      a.propagate()
      b.propagate()
    }
    Sub(a, b) => {
      let a_grad = a.grad
      let b_grad = b.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i]
        b_grad[i] -= self_grad[i]
      }
      a.propagate()
      b.propagate()
    }
    Mul(a, b) => {
      let a_grad = a.grad
      let b_grad = b.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i] * b.value[i]
        b_grad[i] += self_grad[i] * a.value[i]
      }
      a.propagate()
      b.propagate()
    }
    Div(a, b) => {
      let a_grad = a.grad
      let b_grad = b.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i] / b.value[i]
        b_grad[i] -= self_grad[i] * a.value[i] / b.value[i] / b.value[i]
      }
      a.propagate()
      b.propagate()
    }
    Neg(a) => {
      let a_grad = a.grad
      for i in 0..<self_grad.length() {
        a_grad[i] -= self_grad[i]
      }
      a.propagate()
    }
    Exp(a) => {
      let a_grad = a.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i] * self.value[i]
      }
      a.propagate()
    }
    Log(a) => {
      let a_grad = a.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i] / a.value[i]
      }
      a.propagate()
    }
    Abs(a) => {
      let a_grad = a.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i] * (if a.value[i] < 0.0 { -1.0 } else { 1.0 })
      }
      a.propagate()
    }
    Sqrt(a) => {
      let a_grad = a.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i] / ((2.0 : Float) * self.value[i])
      }
      a.propagate()
    }
    Sum(x, m) => {
      for i in 0..<self_grad.length() {
        for j = 0; j < m[i].length(); j = j + 1 {
          x.grad[m[i][j]] += self_grad[i]
        }
      }
      x.propagate()
    }
    Pow(a, b) => {
      let a_grad = a.grad
      for i in 0..<self_grad.length() {
        a_grad[i] = self_grad[i] * b.to_float() * power(a.value[i], b - 1)
      }
      a.propagate()
    }
    MatMul(a, b) => {
      let a_shape = a.shape
      let a_grad = a.grad
      let b_grad = b.grad
      let row = a_shape[0]
      let col = b.shape[1]
      let key = a_shape[1]
      for r = 0; r < row; r = r + 1 {
        for c = 0; c < col; c = c + 1 {
          for k = 0; k < key; k = k + 1 {
            a_grad[r * key + k] += self_grad[r * col + c] * b.value[k * col + c]
            b_grad[k * col + c] += self_grad[r * col + c] * a.value[r * key + k]
          }
        }
      }
      a.propagate()
      b.propagate()
    }
    Get(a, m) => {
      let a_grad = a.grad
      for i in 0..<m.length() {
        a_grad[m[i]] += self_grad[i]
      }
      a.propagate()
    }
    Cat(xs, m) => {
      for i in 0..<self_grad.length() {
        let (x, m) = m[i]
        xs[x].grad[m] += self_grad[i]
      }
      for i in 0..<xs.length() {
        xs[i].propagate()
      }
    }
    Nop(x) => x.propagate()
    ReLU(a) => {
      let a_grad = a.grad
      for i in 0..<self_grad.length() {
        a_grad[i] += self_grad[i] *
          (if self.value[i] > 0.0 { 1.0 } else { 0.0 })
      }
      a.propagate()
    }
    AvgPool2d(x, kernel_size~, stride~) => {
      fn index4d(
        block : @unsafe.UnsafeArray[Int],
        n : Int,
        c : Int,
        h : Int,
        w : Int
      ) -> Int {
        return n * block[0] + c * block[1] + h * block[2] + w
      }

      let iN = self.shape[0]
      let iC = self.shape[1]
      let oH = self.shape[2]
      let oW = self.shape[3]
      let kH = kernel_size.0
      let kW = kernel_size.1
      let kA = (kH * kW).to_float()
      for n = 0; n < iN; n = n + 1 {
        for c = 0; c < iC; c = c + 1 {
          for h = 0; h < oH; h = h + 1 {
            for w = 0; w < oW; w = w + 1 {
              for i = 0; i < kH; i = i + 1 {
                for j = 0; j < kW; j = j + 1 {
                  let i = index4d(
                    x.block,
                    n,
                    c,
                    h * stride.0 + i,
                    w * stride.1 + j,
                  )
                  let o = index4d(self.block, n, c, h, w)
                  x.grad[i] += self_grad[o] / kA
                }
              }
            }
          }
        }
      }
      x.propagate()
    }
    MaxPool2d(x, m) =>
      for i = 0; i < self.value.length(); i = i + 1 {
        x.grad[m[i]] += self_grad[i]
      }
    Correlate2d(image~, kernel~, stride~, padding~) => {
      for i in 0..<self.shape[0] {
        for j in 0..<self.shape[1] {
          let i_start = i * stride.0 - padding.0
          let pi_start = @math.maximum(0, i_start)
          let ki_start = @math.maximum(0, -i_start)
          let i_length = @math.minimum(
            kernel.shape[0] - ki_start,
            image.shape[0] - pi_start,
          )
          let j_start = j * stride.1 - padding.1
          let pj_start = @math.maximum(0, j_start)
          let kj_start = @math.maximum(0, -j_start)
          let j_length = @math.minimum(
            kernel.shape[1] - kj_start,
            image.shape[1] - pj_start,
          )
          for k in 0..<i_length {
            let pj_base = (pi_start + k) * image.shape[1]
            let kj_base = (ki_start + k) * kernel.shape[1]
            for l in 0..<j_length {
              let image_value = image.value[pj_base + pj_start + l]
              let kernel_value = kernel.value[kj_base + kj_start + l]
              image.grad[pj_base + pj_start + l] += self_grad[i * self.shape[1] +
                j] *
                kernel_value
              kernel.grad[kj_base + kj_start + l] += self_grad[i * self.shape[1] +
                j] *
                image_value
            }
          }
        }
      }
      image.propagate()
      kernel.propagate()
    }
  }
}

///|
fn Tensor::check(self : Tensor[Float]) -> Unit {
  if self.refcnt == -1 {
    return
  }
  if self.refcnt != 0 {
    abort("ref != 0")
  }
  match self.graph {
    Val => ()
    Var => ()
    Add(x, y) => {
      x.check()
      y.check()
    }
    Sub(x, y) => {
      x.check()
      y.check()
    }
    Mul(x, y) => {
      x.check()
      y.check()
    }
    Div(x, y) => {
      x.check()
      y.check()
    }
    Neg(x) => x.check()
    Exp(x) => x.check()
    Log(x) => x.check()
    Abs(x) => x.check()
    Sqrt(x) => x.check()
    Sum(x, _) => x.check()
    Pow(x, _) => x.check()
    MatMul(x, y) => {
      x.check()
      y.check()
    }
    Get(a, _) => a.check()
    Cat(xs, _) =>
      for i in 0..<xs.length() {
        xs[i].check()
      }
    Nop(x) => x.check()
    ReLU(x) => x.check()
    AvgPool2d(x, ..) => x.check()
    MaxPool2d(x, _) => x.check()
    Correlate2d(image~, kernel~, ..) => {
      image.check()
      kernel.check()
    }
  }
  self.refcnt = -1
}

///|
pub fn Tensor::backward(self : Tensor[Float]) -> Unit {
  self.incref()
  for i in 0..<self.grad.length() {
    self.grad[i] += 1.0
  }
  self.propagate()
  if @debug.DEBUG {
    self.check()
  }
}

test "Tensor::backward - Add" {
  let a = Tensor::new(1.0)
  let b = Tensor::new(2.0)
  let c = a + b
  c.backward()
  inspect!(a.grad, content="[1]")
  inspect!(b.grad, content="[1]")
}

test "Tensor::backward - Mul" {
  let a = Tensor::new(3.0)
  let b = Tensor::new(4.0)
  let c = a * b
  c.backward()
  inspect!(a.grad, content="[4]")
  inspect!(b.grad, content="[3]")
}

test "Tensor::backward - MatMul" {
  let a = [
    [Tensor::new(1.0), Tensor::new(2.0), Tensor::new(3.0)],
    [Tensor::new(4.0), Tensor::new(5.0), Tensor::new(6.0)],
  ]
  let b = [
    [Tensor::new(7.0), Tensor::new(8.0)],
    [Tensor::new(9.0), Tensor::new(10.0)],
    [Tensor::new(11.0), Tensor::new(12.0)],
  ]
  let c = []
  for i = 0; i < a.length(); i = i + 1 {
    let row = []
    for j = 0; j < b[0].length(); j = j + 1 {
      let mut sum = Tensor::new(0.0)
      for k = 0; k < a[0].length(); k = k + 1 {
        sum = sum + a[i][k] * b[k][j]
      }
      row.push(sum)
    }
    c.push(row)
  }
  inspect!(c, content="[[58, 64], [139, 154]]")
  for row in c {
    for col in row {
      col.backward()
    }
  }
  inspect!(a[0][0].grad, content="[15]")
  inspect!(a[0][1].grad, content="[19]")
  inspect!(a[0][2].grad, content="[23]")
  inspect!(a[1][0].grad, content="[15]")
  inspect!(a[1][1].grad, content="[19]")
  inspect!(a[1][2].grad, content="[23]")
  inspect!(b[0][0].grad, content="[5]")
  inspect!(b[0][1].grad, content="[5]")
  inspect!(b[1][0].grad, content="[7]")
  inspect!(b[1][1].grad, content="[7]")
  inspect!(b[2][0].grad, content="[9]")
  inspect!(b[2][1].grad, content="[9]")
  let x = Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let y = Tensor::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
  let z = x.matmul(y)
  inspect!(z, content="[[58, 64], [139, 154]]")
  z.backward()
  inspect!(x.grad, content="[15, 19, 23, 15, 19, 23]")
  inspect!(y.grad, content="[5, 5, 7, 7, 9, 9]")
}

///|
pub fn Tensor::shape(self : Tensor[Float]) -> FixedArray[Int] {
  self.shape._
}

///|
pub fn Tensor::value(self : Tensor[Float]) -> FixedArray[Float] {
  self.value._
}
