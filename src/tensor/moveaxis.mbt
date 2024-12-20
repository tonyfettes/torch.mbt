///|
pub fn Tensor::moveaxis[Shape : @shape.Shape](
  self : Tensor[Float],
  source : Shape,
  destination : Shape
) -> Tensor[Float] {
  let source = source.to_fixed_array()
  let destination = destination.to_fixed_array()
  if source.length() != destination.length() {
    abort(
      "Source axis (\{source}) and destination axis (\{destination}) must have the same length",
    )
  }
  let permutation = FixedArray::make(self.shape.length(), -1)
  let arranged = FixedArray::make(self.shape.length(), false)
  for i = 0; i < source.length(); i = i + 1 {
    let source = if source[i] < 0 {
      source[i] + self.shape.length()
    } else {
      source[i]
    }
    let destination = if destination[i] < 0 {
      destination[i] + self.shape.length()
    } else {
      destination[i]
    }
    if permutation[destination] != -1 {
      abort(
        "Destination axis \{destination} is already assigned to \{permutation[destination]}",
      )
    }
    permutation[destination] = source
    arranged[source] = true
  }
  let mut dimension = 0
  for i = 0; i < permutation.length(); i = i + 1 {
    if permutation[i] != -1 {
      continue
    }
    while arranged[dimension] {
      dimension += 1
    }
    arranged[dimension] = true
    permutation[i] = dimension
  }
  self.permute(permutation)
}
