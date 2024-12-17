///|
pub(open) trait Module: Show {
  forward(Self, @tensor.Tensor[Float]) -> @tensor.Tensor[Float]
  parameters(Self) -> Iter[@tensor.Tensor[Float]]
}

test "MLP" {
  @distributions.seed(b"12345678223456783234567842345678")
  let layers : Sequential[Module] = Sequential::new([
    Linear::new(4, 8),
    ReLU::new(),
    Linear::new(8, 8),
    ReLU::new(),
    Linear::new(8, 4),
    Softmax::new(),
  ])
  let input : @tensor.Tensor[Float] = @tensor.tensor([1.0, 1.0, 1.0, 1.0])
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.20922499895095825, 0.2580592930316925, 0.25516802072525024, 0.2775477170944214]",
  )
  let target : @tensor.Tensor[Float] = @tensor.tensor([1.0, 0.0, 0.0, 0.0])
  let mut loss : @tensor.Tensor[Float] = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i].to_tensor() * output[i].log()
  }
  inspect!(loss, content="1.5643450021743774")
  let optimizer = @optim.SGD::new(layers.parameters(), learning_rate=0.1)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  let output = layers.forward(input)
  inspect!(
    output,
    content="[0.24395465850830078, 0.2458374947309494, 0.24326206743717194, 0.2669457793235779]",
  )
}

///|
fn forward_[Module : Module](
  tensor : @tensor.Tensor[Float],
  module_ : Module
) -> @tensor.Tensor[Float] {
  module_.forward(tensor)
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
