pub struct Categorical {
  probs : Tensor
}

pub fn Categorical::new(probs : Tensor) -> Categorical {
  Categorical::{ probs, }
}

pub fn Categorical::output(_self : Categorical, logger : Logger) -> Unit {
  logger.write_string("Categorical(probs=")
  Show::output(_self.probs, logger)
  logger.write_string(")")
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
