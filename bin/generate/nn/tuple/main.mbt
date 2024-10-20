let type_names = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
  "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
]

pub fn build_type_parameters(
  length : Int,
  ~constraint : String = "Module"
) -> String {
  let type_parameters = []
  for i in 0..<length {
    type_parameters.push("\{type_names[i]} : \{constraint}")
  }
  String::concat(type_parameters, separator=", ")
}

pub fn build_tuple_type(length : Int) -> String {
  "(" +
  String::concat(type_names[:length].iter().collect(), separator=", ") +
  ")"
}

fn build_forward(length : Int) -> String {
  fn build_body(length : Int) {
    let lines = ["input"]
    for i in 0..<length {
      lines.push("|> forward_(self.\{i})")
    }
    return lines
  }

  let type_parameters = build_type_parameters(length)
  let tuple_type = build_tuple_type(length)
  let body = String::concat(build_body(length), separator="\n  ")
  $|pub impl[\{type_parameters}] Module for \{tuple_type} with forward(
  $|  self : \{tuple_type},
  $|  input : @tensor.Tensor
  $|) -> @tensor.Tensor {
  $|  \{body}
  $|}
}

fn build_parameters(length : Int) -> String {
  fn build_body(length : Int) {
    let lines = ["self.0.parameters()"]
    for i in 1..<length {
      lines.push("|> Iter::concat(self.\{i}.parameters())")
    }
    return lines
  }

  let type_parameters = build_type_parameters(length)
  let tuple_type = build_tuple_type(length)
  let body = String::concat(build_body(length), separator="\n  ")
  $|pub impl[\{type_parameters}] Module for \{tuple_type} with parameters(
  $|  self : \{tuple_type}
  $|) -> Iter[@tensor.Tensor] {
  $|  \{body}
  $|}
}

fn main {
  let functions = []
  for i in 2..=16 {
    functions.push(build_forward(i))
    functions.push(build_parameters(i))
  }
  let content = String::concat(functions, separator="\n\n")
  @fs.write_string_to_file(path="../src/nn/nn_tuple.mbt", ~content)
}
