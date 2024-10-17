// @ts-check
import * as MnistModel from "./mnist-model.js";

/** @typedef {{ type: 'infer', data: MnistModel.Input }} InferMessage */
/** @typedef {{ type: 'train', data: MnistModel.Batch }} TrainMessage */

self.onmessage =
  /** @param {MessageEvent<InferMessage | TrainMessage>} event */
  async (event) => {
    const { type, data } = event.data;
    switch (type) {
      case "infer":
        const result = MnistModel.infer(data);
        self.postMessage({ type: "infer", data: result });
        break;
      case "train":
        const loss = MnistModel.train(data, 0.001);
        self.postMessage({ type: "train", data: loss });
        break;
    }
  };
