///|
pub fn Tensor::to_json[Dtype : ToJson](self : Tensor[Dtype]) -> Json {
  let self_shape = self.shape
  let self_block = self.block()
  let self_value = self.value
  fn build(dimension : Int, offset : Int) -> Json {
    if dimension >= self_shape.length() {
      return self_value[offset].to_json()
    }
    let elements = Array::new(capacity=self_shape[dimension])
    for i = 0; i < self_shape[dimension]; i = i + 1 {
      elements.push(build(dimension + 1, offset + i * self_block[dimension]))
    }
    Json::Array(elements)
  }

  build(0, 0)
}

test "Tensor::to_json" {
  let tensor : Tensor[Float] = tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}

///|
pub fn Tensor::from_json(
  json : Json,
  path : @json.JsonPath
) -> Tensor[Float]!@json.JsonDecodeError {
  match json {
    Json::Number(x) => Tensor::new(x)
    Json::Array(xs) => {
      let value : Array[Tensor[Float]] = Array::new(capacity=xs.length())
      for i, x in xs {
        value.push(Tensor::from_json!(x, @json.add_index(path, i)))
      }
      Tensor::stack(FixedArray::from_array(value))
    }
    _ => raise @json.JsonDecodeError((path, "Expected an array"))
  }
}

test "Tensor::from_json" {
  let json : Json = [[0, 1, 2], [3, 4, 5]]
  let tensor : Tensor[Float] = json |> @json.from_json!()
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}
