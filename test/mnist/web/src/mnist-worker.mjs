// @ts-check
import * as mnist from './mnist.js';

fetch('./mnist-model.json').then((response) => {
  return response.text();
}).then((model) => {
  mnist.load(model);
}).catch(console.error);

self.onmessage =
  /** @param {MessageEvent<{ type: string, data: Float64Array }>} event */
  async (event) => {
    const { data } = event;
    if (data.type === 'infer') {
      const result = mnist.infer(data.data);
      self.postMessage({ type: 'infer', data: result });
    }
  }
