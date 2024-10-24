pub fn Tensor::to_json(self : Tensor) -> Json {
  fn build(dimension : Int, offset : Int) -> Json {
    if dimension >= self.shape.length() {
      return Json::Number(self.value[offset].to_double())
    }
    let elements = Array::new(capacity=self.shape[dimension])
    for i = 0; i < self.shape[dimension]; i = i + 1 {
      elements.push(build(dimension + 1, offset + i * self.block[dimension]))
    }
    Json::Array(elements)
  }

  build(0, 0)
}

test "Tensor::to_json" {
  let tensor = tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}

pub fn Tensor::from_json(
  json : Json,
  path : @json.JsonPath
) -> Tensor!@json.JsonDecodeError {
  match json {
    Json::Number(x) => Tensor::new(x)
    Json::Array(xs) => {
      let value : Array[Tensor] = Array::new(capacity=xs.length())
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
  let tensor : Tensor = json |> @json.from_json!()
  inspect!(tensor, content="[[0, 1, 2], [3, 4, 5]]")
}
