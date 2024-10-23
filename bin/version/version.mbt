let ignore_list = ["target", "node_modules"]

fn should_ignore(item : String) -> Bool {
  if item.starts_with(".") || item.starts_with("_") {
    return true
  }
  ignore_list.contains(item)
}

pub fn update(~path : String, ~version : String) -> Unit! {
  for item in @fs.read_dir!(~path) {
    if should_ignore(item) {
      continue
    }
    let item_path = "\{path}/\{item}"
    if @fs.is_dir!(path=item_path) {
      update!(path=item_path, ~version)
    }
    if item == "moon.mod.json" {
      let content = @fs.read_file_to_string!(path=item_path)
      match @json.parse!(content) {
        Object(
          { "name": String(module_name), "version": String(module_version) }
          as
          module
        ) => {
          if module_version == version {
            continue
          }
          $|\{module_name}: - \{module_version}
          $|\{module_name}: + \{version}
          |> println
          let update : Map[String, Json] = {
            "name": String(module_name),
            "version": String(version),
          }
          for field in module {
            if field.0 != "name" && field.0 != "version" {
              update[field.0] = field.1
            }
          }
          let content = @json.stringify(Object(update), indent=2)
          @fs.write_string_to_file(path=item_path, content=content + "\n")
        }
        Object({ "name": String(module_name) } as module) => {
          let update : Map[String, Json] = {
            "name": String(module_name),
            "version": String(version),
          }
          for field in module {
            if field.0 != "name" && field.0 != "version" {
              update[field.0] = field.1
            }
          }
          $|\{module_name}: + \{version}
          |> println
          let content = @json.stringify(Object(update), indent=2)
          @fs.write_string_to_file(path=item_path, content=content + "\n")
        }
        _ =>
          $|Module at \{item_path} has no name/version, or they are not string
          |> println
      }
    }
  }
}

fn main {
  try {
    let module = @fs.read_file_to_string!(path="../moon.mod.json")
      |> @json.parse!()
    match module {
      { "name": String(name), "version": String(version) } => {
        $|\{name}: \{version}
        |> println
        update!(path="..", ~version)
      }
      _ => println("moon.mod.json has no name/version, or they are not string")
    }
  } catch {
    error => println("Error: \{Show::to_string(error)}")
  }
}
