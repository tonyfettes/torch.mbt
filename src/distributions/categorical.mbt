struct Categorical {
  probs : @tensor.Tensor
  logits : @tensor.Tensor
}

pub fn Categorical::of_probs(probs : @tensor.Tensor) -> Categorical {
  let logits = probs.log()
  Categorical::{ probs, logits }
}

pub fn Categorical::of_logits(logits : @tensor.Tensor) -> Categorical {
  let probs = {
    let exp = logits.exp()
    let sum = exp.sum()
    exp / sum
  }
  Categorical::{ probs, logits }
}

pub fn Categorical::sample(
  self : Categorical,
  shape : @shape.Shape
) -> @tensor.Tensor {
  let batch_shape = []
  for i in 0..<(self.probs.shape.length() - 1) {
    batch_shape.push(self.probs.shape[i])
  }
  let batch_shape = batch_shape.append([..shape])
  let sample = @tensor.zeros(batch_shape)
  self.sample_(sample)
  sample
}

pub fn Categorical::sample_(
  self : Categorical,
  sample : @tensor.Tensor
) -> Unit {
  let probs = self.probs
  let width = probs.shape[probs.shape.length() - 1]
  fn sample_(d : Int, p : Int) -> Unit {
    if d == probs.shape.length() - 1 {
      let mut choice = rand.val.float()
      for i in 0..<width {
        choice = choice - probs.value[p + i]
        if choice <= 0.0 {
          sample.value[p] = i.to_float()
          return
        }
      }
      sample.value[p] = rand.val.int(limit=width).to_float()
    }
    for i in 0..<probs.shape[d] {
      sample_(d + 1, p + i * probs.block[d])
    }
  }

  sample_(0, 0)
}

pub fn Categorical::log_prob(
  self : Categorical,
  value : @tensor.Tensor
) -> @tensor.Tensor {
  @tensor.gather(
    self.logits,
    value.shape.length() - 1,
    value.reshape([..value.shape, 1]),
  )
}
