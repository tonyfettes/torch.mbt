pub trait Continuous {
  sample(Self, ~shape: Array[Int]) -> Tensor
}

pub trait Discrete {
  sample(Self) -> Int
}

impl Continuous for Double with sample(self : Double, ~shape : Array[Int]) -> Tensor {
  let mut size = 1
  for s in shape {
    size *= s
  }
  let sample = []
  for i = 0; i < size; i = i + 1 {
    sample.push(self)
  }
  Tensor::new(sample).reshape(shape)
}

pub struct Uniform {
  lower : Double
  upper : Double
}

pub fn Uniform::new(lower : Double, upper : Double) -> Uniform {
  Uniform::{ lower, upper }
}

fn Uniform::_sample(self : Uniform) -> Double {
  rand.val.double() * (self.upper - self.lower) + self.lower
}

pub fn Uniform::sample(self : Uniform, ~shape : Array[Int]) -> Tensor {
  let mut size = 1
  for s in shape {
    size *= s
  }
  let sample = []
  for i = 0; i < size; i = i + 1 {
    sample.push(self._sample())
  }
  Tensor::new(sample).reshape(shape)
}

let normal_buffer : Array[Double] = []

pub fn normal(~mean : Double = 0.0, ~std : Double = 1.0) -> Double {
  match normal_buffer.pop() {
    None => ()
    Some(z1) => return z1
  }
  let u1 = rand.val.double()
  let u2 = rand.val.double()
  let r = (-2.0 * u1.ln()).sqrt()
  let z0 = r * @math.cos(2.0 * pi * u2)
  let z1 = r * @math.sin(2.0 * pi * u2)
  normal_buffer.push(z1)
  return z0 * std + mean
}

pub struct Normal {
  mean : Double
  std : Double
}

let pi = 0x3.243F6A8885A308CA8A54

pub fn Normal::new(~mean : Double = 0.0, ~std : Double = 1.0) -> Normal {
  Normal::{ mean, std }
}

pub fn Normal::_sample(self : Normal) -> Double {
  normal(mean=self.mean, std=self.std)
}

pub fn Normal::sample(self : Uniform, ~shape : Array[Int]) -> Tensor {
  let mut size = 1
  for s in shape {
    size *= s
  }
  let sample = []
  for i = 0; i < size; i = i + 1 {
    sample.push(self._sample())
  }
  Tensor::new(sample).reshape(shape)
}

pub struct Categorical {
  logits : Tensor
  probs : Tensor
} derive(Show)

pub fn Categorical::new(probs : Tensor) -> Categorical {
  let logits = probs.log()
  Categorical::{ logits, probs }
}

pub fn Categorical::sample(self : Categorical) -> Int {
  if self.probs.shape.length() != 1 {
    abort("Categorical distribution only supports 1D probabilities")
  }
  let mut choice = rand.val.double()
  for index, value in self.probs.value {
    choice = choice - value
    if choice <= 0.0 {
      return index
    }
  }
  rand.val.int(limit=self.probs.length())
}

pub fn Categorical::log_prob(self : Categorical, sample : Int) -> Tensor {
  self.logits[sample].to_tensor()
}
