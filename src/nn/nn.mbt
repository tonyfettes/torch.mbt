pub trait Module: Show {
  forward(Self, @tensor.Tensor) -> @tensor.Tensor
  parameters(Self) -> Iter[@tensor.Tensor]
}

test "MLP" {
  @distributions.seed(b"12345678223456783234567842345678")
  let layers : Sequential[Module] = Sequential::new(
    [
      Linear::new(4, 8),
      ReLU::new(),
      Linear::new(8, 8),
      ReLU::new(),
      Linear::new(8, 4),
      Softmax::new(),
    ],
  )
  let input = @tensor.tensor([1.0, 1.0, 1.0, 1.0])
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.20922499162142535, 0.25805928787959304, 0.2551680112861502, 0.27754770921283145]",
  )
  let target = @tensor.tensor([1.0, 0.0, 0.0, 0.0])
  let mut loss = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i].to_tensor() * output[i].log()
  }
  inspect!(loss, content="1.5643450911089625")
  let optimizer = @optim.SGD::new(layers.parameters(), learning_rate=0.1)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.24395465491177973, 0.24583749392308907, 0.24326207530914587, 0.2669457758559854]",
  )
}

pub trait Loss {
  forward(Self, @tensor.Tensor, @tensor.Tensor) -> @tensor.TensorValue
}

fn forward_[Module : Module](
  tensor : @tensor.Tensor,
  module : Module
) -> @tensor.Tensor {
  module.forward(tensor)
}

// pub fn transpile[Module : Module](
//   module : Module,
//   export : String
// ) -> String {
//   let transpiler = @transpile.PyTorchTranspiler::new()
//   let variable = @transpile.ToPyTorchSource::transpile(module, transpiler)
//   transpiler.export(variable, export)
//   transpiler.to_string()
// }
