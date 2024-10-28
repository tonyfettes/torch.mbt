struct Data {
  image : Array[Int]
  label : Int
} derive(FromJson)

struct Dataset {
  train : Array[Data]
  eval : Array[Data]
}

fn Dataset::load(path : String) -> Dataset! {
  let train = []
  for i = 0; i < 60000; i = i + 1 {
    let data : Data = @fs.read_file_to_string!(path="\{path}/train_\{i}.json")
      |> @json.parse!()
      |> @json.from_json!()
    train.push(data)
  }
  let eval = []
  for i = 0; i < 10000; i = i + 1 {
    let data : Data = @fs.read_file_to_string!(path="\{path}/test_\{i}.json")
      |> @json.parse!()
      |> @json.from_json!()
    eval.push(data)
  }
  Dataset::{ train, eval }
}

fn eval(model : @torch/nn.Module, dataset : Array[Data]) -> Double {
  let mut accuracy = 0.0
  for datum in dataset {
    let input = @torch.tensor(datum.image.map(fn(x) { x.to_double() / 255.0 })).reshape(
      [28, 28],
    )
    let target = Array::make(10, 0.0)
    target[datum.label] = 1.0
    let target = @torch.tensor(target)
    let softmax = @torch/nn.Softmax::new()
    let output = softmax.forward(model.forward(@torch.stack([[input]])))
    let value = (output * target).sum().to_double()
    accuracy += value
  }
  let accuracy = accuracy / dataset.length().to_double()
  accuracy
}

fn train[Module : @torch/nn.Module](
  model : Module,
  dataset : Array[Data]
) -> Unit {
  let criterion = @torch/nn.CrossEntropyLoss::new()
  let optimizer = @torch/optim.SGD::new(model.parameters(), learning_rate=0.01)
  let mut step = 0
  let batch = 10
  for epoch = 0; epoch < 3; epoch = epoch + 1 {
    for i = 0; i < dataset.length(); i = i + batch {
      let input = FixedArray::makei(
        batch,
        fn(b) {
          let data = dataset[i + b]
          @torch.tensor(data.image.map(fn(x) { x.to_double() / 255.0 })).reshape(
            [1, 28, 28],
          )
        },
      )
      let input = @torch.stack(input)
      let target = FixedArray::makei(
        batch,
        fn(b) {
          let target = FixedArray::make(10, 0.0)
          target[dataset[i + b].label] = 1.0
          target
        },
      )
      let target = @torch.tensor(target)
      let output = model.forward(input)
      optimizer.zero_grad()
      let loss = criterion.forward(output, target)
      println("step: \{step}, loss: \{loss}")
      loss.backward()
      optimizer.step()
      step += 1
    }
  }
}

fn main {
  let model = @mnist.Model::new()
  let dataset = try {
    Dataset::load!("data")
  } catch {
    error => {
      let buffer = StringBuilder::new()
      Show::output(error, buffer)
      abort("error: \{buffer.to_string()}")
    }
  }
  // let accuracy = eval(model, dataset.eval)
  // println("accuracy: \{accuracy}")
  let train_example = dataset.train[0]
  println("train_example: \{train_example.image}")
  train(model, dataset.train)
  let accuracy = eval(model, dataset.eval)
  println("accuracy: \{accuracy}")
  @fs.write_string_to_file(
    path="model.json",
    content=model.to_json().stringify(),
  )
}
