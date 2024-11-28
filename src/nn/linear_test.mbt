test "Linear" {
  let linear = @nn.Linear::new(2, 2)
  @tensor.with_no_grad(
    fn() {
      linear.weight[0][0] = 1.0
      linear.weight[0][1] = 3.0
      linear.weight[1][0] = 2.0
      linear.weight[1][1] = 4.0
      linear.bias[0] = 5.0
      linear.bias[1] = 6.0
    },
  )
  let input = @tensor.tensor([1.0, 2.0])
  let optimizer = @optim.SGD::new(linear.parameters(), learning_rate=0.1)
  let output = linear.forward(input)
  inspect!(output, content="[10, 17]")
  let target = @tensor.tensor([16.0, 16.0])
  let mut loss = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss += output[i] - target[i]
  }
  inspect!(loss, content="59")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  inspect!(
    linear.weight,
    content="[[0.8999999761581421, 2.9000000953674316], [1.7999999523162842, 3.799999952316284]]",
  )
  inspect!(linear.bias, content="[4.900000095367432, 5.900000095367432]")
  let output = linear.forward(input)
  inspect!(output, content="[9.399999618530273, 16.399999618530273]")
}

test "Linear with ReLU" {
  let linear = @nn.Linear::new(2, 2)
  @tensor.with_no_grad(
    fn() {
      linear.weight[0][0] = 1.0
      linear.weight[0][1] = 3.0
      linear.weight[1][0] = 2.0
      linear.weight[1][1] = 4.0
      linear.bias[0] = 5.0
      linear.bias[1] = 6.0
    },
  )
  let layers : @nn.Sequential[@nn.Module] = @nn.Sequential::new(
    [linear, @nn.ReLU::new()],
  )
  let optimizer = @optim.SGD::new(linear.parameters(), learning_rate=1)
  let input = @tensor.tensor([2.0, 2.0])
  let output = layers.forward(input)
  inspect!(output, content="[11, 20]")
  let target = @tensor.tensor([1.0, 0.0])
  let mut loss = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i].to_tensor() * output[i].log()
  }
  inspect!(loss, content="-2.397895336151123")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  inspect!(
    linear.weight,
    content="[[1.1818182468414307, 3], [2.1818182468414307, 4]]",
  )
  let output = layers.forward(input)
  inspect!(output, content="[11.818181991577148, 20]")
}

test "Linear with Softmax" {
  let linear = @nn.Linear::new(2, 2)
  @tensor.with_no_grad(
    fn() {
      linear.weight[0][0] = 2.0
      linear.weight[0][1] = 2.0
      linear.weight[1][0] = 2.0
      linear.weight[1][1] = 2.0
      linear.bias[0] = 1.0
      linear.bias[1] = 2.0
    },
  )
  let layers : @nn.Sequential[@nn.Module] = @nn.Sequential::new(
    [linear, @nn.Softmax::new()],
  )
  let optimizer = @optim.SGD::new(linear.parameters(), learning_rate=1)
  let input = @tensor.tensor([2.0, 2.0])
  let output = layers.forward(input)
  inspect!(output, content="[0.2689414322376251, 0.7310585975646973]")
  let target = @tensor.tensor([1.0, 0.0])
  let mut loss = @tensor.tensor(0.0)
  for i = 0; i < output.length(); i = i + 1 {
    loss = loss - target[i].to_tensor() * output[i].log()
  }
  inspect!(loss, content="1.31326162815094")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  let output = layers.forward(input)
  inspect!(output, content="[0.9999947547912598, 0.0000052406817303563]")
}

test "Linear Regression" {
  let model = @nn.Linear::new(1, 1)
  @tensor.with_no_grad(
    fn() {
      model.weight[0][0] = 0.0
      model.bias[0] = 0.0
    },
  )
  let optimizer = @optim.SGD::new(model.parameters(), learning_rate=0.1)
  let target = fn(x : Array[Double]) { [2.0 * x[0] + 3] }
  let inputs = [
    0.8208251, 0.90000966, 0.28106993, 0.39323831, 0.43506248, 0.4969614, 0.52247973,
    0.49798604, 0.43939068, 0.3734946, 0.4441855, 0.17318048, 0.53219596, 0.63448068,
    0.27115805, 0.04605697, 0.62318401, 0.33910485, 0.69114928, 0.85872608, 0.70759624,
    0.40058802, 0.1777967, 0.92748176, 0.94344838, 0.78039178, 0.33418037, 0.43679556,
    0.39154924, 0.10587376, 0.05049539, 0.29797588, 0.00861857, 0.40571597, 0.21740392,
    0.65157348, 0.08771972, 0.72849501, 0.08581929, 0.8128746, 0.18798559, 0.30142073,
    0.32481787, 0.52926784, 0.1449449, 0.20553479, 0.38707897, 0.40028778, 0.11176918,
    0.76972532, 0.42597432, 0.01741988, 0.80524058, 0.35739401, 0.24440017, 0.41526613,
    0.99033731, 0.81271161, 0.46265435, 0.95218477, 0.62453272, 0.15694713, 0.20849319,
    0.97591847, 0.48845406, 0.72073021, 0.52854909, 0.41928837, 0.40103961, 0.66694493,
    0.79072028, 0.19673913, 0.45183849, 0.02185975, 0.15932374, 0.89000227, 0.06060862,
    0.45457933, 0.86395899, 0.54407434, 0.36498587, 0.33118886, 0.97118758, 0.85580425,
    0.53489337, 0.71204262, 0.72440047, 0.63693202, 0.57291731, 0.04470077, 0.74890034,
    0.17283568, 0.01892377, 0.88189203, 0.5364349, 0.56212932, 0.68814933, 0.02615991,
    0.98899348, 0.25727347,
  ]
  for input in inputs {
    let model_output = model.forward(@tensor.tensor([input]))
    let target_output = target([input])
    let mse = @nn.MSELoss::new()
    let loss = mse.forward(model_output, @tensor.tensor(target_output))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  }
  inspect!(model.weight[0][0], content="1.9810588359832764")
  inspect!(model.bias[0], content="3.0104925632476807")
}

test "linear/consistent" (it : @test.T) {
  @distributions.seed(b"12345678223456783234567842345678")
  let distribution = @distributions.Uniform::new(-1.0, 1.0)
  let model = @nn.Linear::new(8, 16)
  let input = distribution.sample([1, 8])
  @json.inspect!(
    input,
    content=[
      [
        -0.4301600456237793, 0.7356255054473877, 0.7850341796875, 0.32139527797698975,
        0.6747739315032959, 0.8503944873809814, 0.2769254446029663, -0.5336059331893921,
      ],
    ],
  )
  let output = model.forward(input)
  inspect!(input.shape, content="[1, 8]")
  inspect!(model.weight.shape, content="[8, 16]")
  let input_weight = input.matmul(model.weight)
  inspect!(input_weight.shape, content="[1, 16]")
  @json.inspect!(
    input_weight,
    content=[
      [
        -0.14556150138378143, 0.34053102135658264, -0.2727360129356384, 0.09315243363380432,
        0.2314411848783493, 0.009322114288806915, -0.22513717412948608, 0.004441825672984123,
        -0.35770153999328613, 0.35220563411712646, 0.015071094036102295, -0.3460395932197571,
        0.28992030024528503, 0.3369879126548767, -0.3778506815433502, 0.22615066170692444,
      ],
    ],
  )
  it.writeln(
    #|import torch
    #|
    #|def test_linear():
    $|    input = torch.tensor(\{input.to_json().stringify()}, dtype=torch.float32)
    $|    weight = torch.tensor(\{model.weight.to_json().stringify()}, dtype=torch.float32)
    $|    bias = torch.tensor(\{model.bias.to_json().stringify()}, dtype=torch.float32)
    $|    output = torch.nn.functional.linear(input, weight.transpose(-2, -1), bias)
    $|    assert torch.allclose(output, torch.tensor(\{output.to_json().stringify()}, dtype=torch.float32))
    ,
  )
  it.snapshot!(filename="nn_linear_test.py")
}
