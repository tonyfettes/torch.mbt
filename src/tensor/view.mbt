enum TensorIndex {
  Index(Int)
  Slice(Int, Int)
} derive(Show)

pub(readonly) struct TensorView {
  tensor : Tensor
  shape : FixedArray[Int]
  index : FixedArray[TensorIndex]
}

pub fn TensorView::output(self : TensorView, logger : Logger) -> Unit {
  let shape = self.tensor.shape
  let block = self.tensor.block
  let value = self.tensor.value
  fn build(d : Int, p : Int) {
    if d == shape.length() {
      Show::output(value[p], logger)
      return
    }
    let (start, end) = if d < self.index.length() {
      match self.index[d] {
        Index(index) => {
          build(d + 1, p + index * block[d])
          return
        }
        Slice(start, end) => (start, end)
      }
    } else {
      (0, shape[d])
    }
    logger.write_char('[')
    for i in start..<end {
      build(d + 1, p + i * block[d])
      if i < end - 1 {
        logger.write_string(", ")
      }
    }
    logger.write_char(']')
  }

  build(0, 0)
}

pub fn TensorView::to_string(self : TensorView) -> String {
  Show::to_string(self)
}

pub fn TensorView::to_json(self : TensorView) -> Json {
  let shape = self.tensor.shape
  let dimension = shape.length()
  let block = self.tensor.block
  let value = self.tensor.value
  fn build(d : Int, p : Int) {
    if d == dimension {
      return Json::Number(value[p].to_double())
    }
    let (start, end) = if d < self.index.length() {
      match self.index[d] {
        Index(index) => return build(d + 1, p + index * block[d])
        Slice(start, end) => (start, end)
      }
    } else {
      (0, shape[d])
    }
    let elements = Array::new(capacity=end - start)
    for i in start..<end {
      elements.push(build(d + 1, p + i * block[d]))
    }
    Json::Array(elements)
  }

  build(0, 0)
}

pub fn TensorView::op_get(self : TensorView, index : Int) -> TensorView {
  if self.index.length() == self.tensor.shape.length() {
    abort("Taking element at the lowest dimension")
  }
  if index >= self.tensor.shape[self.index.length()] {
    abort(
      "Tensor element access at \{index} out of bounds [0, \{self.tensor.shape[self.index.length()]}",
    )
  }
  let shape = FixedArray::from_iter(self.shape.iter().drop(1))
  let index = FixedArray::from_iter(self.index.iter().append(Index(index)))
  TensorView::{ tensor: self.tensor, shape, index }
}

pub fn TensorView::op_set(
  self : TensorView,
  index : Int,
  other : ToTensor
) -> Unit {
  if not(no_grad.val) {
    match self.tensor.graph {
      Val => ()
      _ =>
        abort(
          "a view of a leaf Variable that requires grad is being used in an in-place operation.",
        )
    }
  }
  let other = other.to_tensor()
  let index = [..self.index, Index(index)]
  let shape = self.tensor.shape
  let block = self.tensor.block
  let value = self.tensor.value
  fn assign(self_dim : Int, self_pos : Int, other_dim : Int, other_pos : Int) {
    if self_dim == shape.length() {
      if other_dim != other.shape.length() {
        abort(
          "Dimension does not match: expecting \{other_dim}, got \{other.shape.length()}",
        )
      }
      value[self_pos] = other.value[other_pos]
      return
    }
    let (start, end) = if self_dim < index.length() {
      match index[self_dim] {
        Index(index) => {
          assign(
            self_dim + 1,
            self_pos + index * block[self_dim],
            other_dim,
            other_pos,
          )
          return
        }
        Slice(start, end) => (start, end)
      }
    } else {
      (0, shape[self_dim])
    }
    if end - start != other.shape[other_dim] {
      abort(
        "\{other_dim}-th dimension should have length \{end - start}, got \{other.shape[other_dim]}",
      )
    }
    for i in start..<end {
      assign(
        self_dim + 1,
        self_pos + i * block[self_dim],
        other_dim + 1,
        other_pos + (i - start) * other.block[other_dim],
      )
    }
  }

  assign(0, 0, 0, 0)
}

test "TensorView::op_set" {
  let tensor = Tensor::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  tensor[:][1] = [7.0, 8.0]
  inspect!(tensor, content="[[1, 7, 3], [4, 8, 6]]")
  tensor[:][0] = [0.0, 1.0]
  inspect!(tensor, content="[[0, 7, 3], [1, 8, 6]]")
  tensor[:][2] = [2.0, 3.0]
  inspect!(tensor, content="[[0, 7, 2], [1, 8, 3]]")
  tensor[0][0] = 9.0
  inspect!(tensor, content="[[9, 7, 2], [1, 8, 3]]")
}

pub fn TensorView::op_as_view(
  self : TensorView,
  start~ : Int,
  end? : Int
) -> TensorView {
  let end = match end {
    None => self.tensor.shape[self.index.length()]
    Some(end) => end
  }
  let shape = self.shape.copy()
  shape[0] = end - start
  let index = FixedArray::from_iter(self.index.iter().append(Slice(start, end)))
  TensorView::{ tensor: self.tensor, shape, index }
}

pub fn TensorView::to_tensor(self : TensorView) -> Tensor {
  let shape : Array[Int] = Array::new(capacity=self.index.length())
  for part in self.index {
    match part {
      Index(_) => ()
      Slice(start, stop) => shape.push(stop - start)
    }
  }
  for i = self.index.length(); i < self.tensor.shape.length(); i = i + 1 {
    shape.push(self.tensor.shape[i])
  }
  // We can't avoid this allocation because we don't know the length of the
  // array.
  let shape = FixedArray::from_array(shape)
  let block = compute_block_size(shape)
  let total = if shape.length() == 0 { 1 } else { shape[0] * block[0] }
  let value : FixedArray[Float] = FixedArray::make(total, 0.0)
  let mut value_length = 0
  let index = FixedArray::make(total, 0)
  let mut index_length = 0
  fn build(dimension : Int, offset : Int) -> Unit {
    if dimension >= self.tensor.shape.length() {
      value[value_length] = self.tensor.value[offset]
      value_length += 1
      index[index_length] = offset
      index_length += 1
      return
    }
    if dimension >= self.index.length() {
      for i in 0..<self.tensor.shape[dimension] {
        build(dimension + 1, offset + i * self.tensor.block[dimension])
      }
      return
    }
    match self.index[dimension] {
      Index(index) =>
        build(dimension + 1, offset + index * self.tensor.block[dimension])
      Slice(start, end) =>
        for i in start..<end {
          build(dimension + 1, offset + i * self.tensor.block[dimension])
        }
    }
  }

  build(0, 0)
  Tensor::{
    value,
    shape,
    block,
    graph: Graph::Get(self.tensor, index),
    refcnt: 0,
    grad: FixedArray::make(total, (0.0 : Float)),
  }
}

test "TensorView::to_tensor" {
  let tensor = Tensor::new([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  let view = tensor[1]
  let tensor = view.to_tensor()
  inspect!(tensor.value, content="[3, 4, 5]")
  inspect!(tensor.shape, content="[3]")
  inspect!(tensor.block, content="[1]")
  inspect!(tensor.graph, content="Get([[0, 1, 2], [3, 4, 5]], [3, 4, 5])")
  inspect!(tensor.refcnt, content="0")
  inspect!(tensor.grad, content="[0, 0, 0]")
  inspect!(tensor, content="[3, 4, 5]")
}

pub fn TensorView::to_double(self : TensorView) -> Double {
  self.to_tensor().to_double()
}

pub fn TensorView::op_add(self : TensorView, other : TensorView) -> Tensor {
  self.to_tensor() + other.to_tensor()
}

pub fn TensorView::op_sub(self : TensorView, other : TensorView) -> Tensor {
  self.to_tensor() + other.to_tensor()
}

pub fn TensorView::op_mul(self : TensorView, other : TensorView) -> Tensor {
  self.to_tensor() * other.to_tensor()
}

pub fn TensorView::log(self : TensorView) -> Tensor {
  self.to_tensor().log()
}

pub fn TensorView::iter(self : TensorView) -> Iter[TensorView] {
  let shape = self.shape
  let dimension = self.shape.length()
  if dimension == 0 {
    abort("Iterating over a scalar tensor is not allowed.")
  }
  Iter::new(
    fn(yield_) {
      for i in 0..<shape[0] {
        let shape = FixedArray::make(dimension - 1, 0)
        for j in 1..<dimension {
          shape[j - 1] = self.shape[j]
        }
        let index : FixedArray[TensorIndex] = FixedArray::make(
          self.index.length() + 1,
          self.index[0],
        )
        for j in 0..<self.index.length() {
          index[j] = self.index[j]
        }
        index[self.index.length()] = Index(i)
        let view = TensorView::{ tensor: self.tensor, shape, index }
        match yield_(view) {
          IterContinue => ()
          IterEnd => break IterEnd
        }
      } else {
        IterEnd
      }
    },
  )
}
