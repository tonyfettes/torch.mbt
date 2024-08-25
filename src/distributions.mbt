pub trait Continuous {
  sample(Self) -> Double
}

pub trait Discrete {
  sample(Self) -> Int
}

impl Continuous for Double with sample(self : Double) -> Double {
  self
}

pub struct Uniform {
  lower : Double
  upper : Double
}

pub fn Uniform::new(lower : Double, upper : Double) -> Uniform {
  Uniform::{ lower, upper }
}

pub fn Uniform::sample(self : Uniform) -> Double {
  rand.double() * (self.upper - self.lower) + self.lower
}

let normal_buffer : Array[Double] = []

pub fn normal(~mean : Double = 0.0, ~std : Double = 1.0) -> Double {
  match normal_buffer.pop() {
    None => ()
    Some(z1) => return z1
  }
  let u1 = rand.double()
  let u2 = rand.double()
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

pub fn Normal::sample(self : Normal) -> Double {
  normal(mean=self.mean, std=self.std)
}

pub struct Categorical {
  logits : Tensor
  probs : Tensor
} derive(Show)

pub fn Categorical::new(probs : Tensor) -> Categorical {
  let logits = probs.map(fn(p) { p.log() })
  Categorical::{ logits, probs }
}

pub fn Categorical::sample(self : Categorical) -> Int {
  let mut choice = rand.double()
  for index, value in self.probs.0 {
    choice = choice - value.value
    if choice <= 0.0 {
      return index
    }
  }
  rand.int(limit=self.probs.length())
}

pub fn Categorical::log_prob(self : Categorical, sample : Int) -> Value {
  self.logits[sample]
}
