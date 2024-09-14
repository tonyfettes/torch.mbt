pub fn entry(canvas : @canvas.Canvas2D) -> Unit {
  let mut last : (Double, Double)? = None
  let model : Model = try {
    model |> @json.parse!() |> @json.from_json!()
  } catch {
    error => {
      let buffer = Buffer::new()
      Show::output(error, buffer)
      abort("error: \{buffer.to_string()}")
    }
  }
  canvas.set_stroke_style(Color("black"))
  canvas.set_line_width(2)
  @canvas.set_on_mouse_down(
    fn(event) {
      let offset_x = event.offset_x() / 10
      let offset_y = event.offset_y() / 10
      last = Some((offset_x, offset_y))
    },
  )
  @canvas.set_on_mouse_move(
    fn(event) {
      if event.buttons().land(1) == 1 {
        let offset_x = event.offset_x() / 10
        let offset_y = event.offset_y() / 10
        match last {
          Some((x, y)) => {
            canvas.begin_path()
            canvas.move_to(x, y)
            canvas.line_to(offset_x, offset_y)
            canvas.stroke()
            last = Some((offset_x, offset_y))
            let image_data = canvas.get_image_data(0, 0, 28, 28)
            let data = image_data.get_data()
            let input = []
            for i = 0; i < 28; i = i + 1 {
              let row = []
              for j = 0; j < 28; j = j + 1 {
                row.push(data[(i * 28 + j) * 4 + 3].to_int())
              }
              println(row)
            }
            for i = 0; i < data.length(); i = i + 4 {
              let a = data[i + 3].to_int()
              input.push(a.to_double() / 255.0)
            }
            println(input)
            let output = model.forward(@torch.tensor(input))
            println(output)
            let mut label = 0
            let mut value = 0.0
            for i = 0; i < output.value.length(); i = i + 1 {
              if output.value[i] > value {
                label = i
                value = output.value[i]
              }
            }
            println("label: \{label}, value: \{value}")
          }
          None => last = Some((offset_x, offset_y))
        }
      } else {
        last = None
      }
    },
  )
}
