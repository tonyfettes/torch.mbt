import * as mnist from './mnist.js';
import model from './mnist-model.json' with { type: 'json' };

mnist.load(JSON.stringify(model));

self.onmessage = async (event) => {
  const { data } = event;
  if (data.type === 'infer') {
    console.log("data", data);
    const result = mnist.infer(data.data);
    self.postMessage({ type: 'infer', data: result });
  }
}
