fn main {
  @fs.write_string_to_file(
    path="cli/mnist/refer.py",
    content=@transpile.transpile(@mnist.Model::new(), "model"),
  )
}
