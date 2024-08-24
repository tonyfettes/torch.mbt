test "Value::backward - op_neg" {
  let v1 = @torch.Value::var(5.0)
  let neg = -v1
  inspect!(neg, content="-5.0")
  neg.backward(1.0)
  inspect!(v1, content="6.0")
}

test "Value::backward - op_add" {
  let v1 = @torch.Value::var(5.0)
  let v2 = @torch.Value::var(4.0)
  let sum = v1 + v2
  inspect!(sum, content="9.0")
  sum.backward(1.0)
  inspect!(v1, content="4.0")
  inspect!(v2, content="3.0")
}

test "Value::backward - op_mul" {
  let v1 = @torch.Value::var(5.0)
  let v2 = @torch.Value::var(4.0)
  let product = v1 * v2
  inspect!(product, content="20.0")
  product.backward(1.0)
  inspect!(v1, content="1.0")
  inspect!(v2, content="-1.0")
}

test "Value::backward - op_div" {
  let v1 = @torch.Value::var(5.0)
  let v2 = @torch.Value::var(4.0)
  let quotient = v1 / v2
  inspect!(quotient, content="1.25")
  quotient.backward(1.0)
  inspect!(v1, content="4.75")
  inspect!(v2, content="4.3125")
}

test "Value::backward - exp" {
  let v1 = @torch.Value::var(5.0)
  let exp = v1.exp()
  inspect!(exp, content="148.4131591025766")
  for i = 0; i < 10; i = i + 1 {
    exp.backward(0.01)
  }
  inspect!(v1, content="-9.84131591025766")
}

test "Value::backward - log" {
  let v1 = @torch.Value::var(5.0)
  let log = v1.log()
  inspect!(log, content="1.6094379124341003")
  log.backward(1.0)
  inspect!(v1, content="4.8")
}

test "Value::backward - sum" {
  let v = [
    @torch.Value::var(1.0),
    @torch.Value::var(2.0),
    @torch.Value::var(3.0),
  ]
  let sum = @torch.Value::sum(v)
  inspect!(sum, content="6.0")
  sum.backward(1.0)
  inspect!(v[0], content="0.0")
  inspect!(v[1], content="1.0")
  inspect!(v[2], content="2.0")
}

test "Value::backward - pow" {
  let v1 = @torch.Value::var(5.0)
  let square = v1.pow(2)
  inspect!(square, content="25.0")
  square.backward(0.1)
  inspect!(v1, content="4.0")
}

test "Linear Regression" {
  let k = @torch.Value::var(0.0)
  let b = @torch.Value::var(0.0)
  let target = fn(x : Double) { 2.0 * x + 3.0 }
  let model = fn(x : Double) { k * @torch.Value::val(x) + b }
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
  for x in inputs {
    let target_value = target(x)
    let model_value = model(x)
    let loss = (@torch.Value::val(target_value) - model_value).pow(2)
    loss.backward(0.1)
  }
  inspect!(k, content="1.981059015531822")
  inspect!(b, content="3.01049253711016")
}
