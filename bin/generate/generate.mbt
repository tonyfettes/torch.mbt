///|
fn remove_whitespace(content : String) -> String {
  let result = StringBuilder::new()
  for c in content {
    if c != ' ' && c != '\n' && c != '\t' {
      result.write_char(c)
    }
  }
  result.to_string()
}

///|
pub fn write_on_change(path~ : String, content~ : String) -> Unit! {
  if @fs.path_exists(path~) {
    let old_no_space = @fs.read_file_to_string!(path~) |> remove_whitespace
    let new_no_space = remove_whitespace(content)
    if old_no_space == new_no_space {
      return
    }
  }
  @fs.write_string_to_file(path~, content~)
}
