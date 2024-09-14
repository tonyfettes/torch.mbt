// @ts-check
import * as mnist from './mnist.js';
import model from './mnist-model.json' with { type: 'json' };

mnist.load(JSON.stringify(model));

self.onmessage =
  /** @param {MessageEvent<{ type: string, data: Float64Array }>} event */
  async (event) => {
    const { data } = event;
    if (data.type === 'infer') {
      const result = mnist.infer(data.data);
      self.postMessage({ type: 'infer', data: result });
    }
  }
