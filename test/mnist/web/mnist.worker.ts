import * as MnistModel from "./mnist-model";

export type MnistWorkerRequest =
  | {
      type: "infer";
      data: Float64Array;
    }
  | {
      type: "train";
      data: {
        batch: [Float64Array, label: number][];
        learningRate: number;
      };
    };

export type MnistWorkerResponse =
  | {
      type: "infer";
      data: Float64Array;
    }
  | {
      type: "train";
      data: number;
    };

self.addEventListener("message", (event: MessageEvent<MnistWorkerRequest>) => {
  const { type, data } = event.data;
  switch (type) {
    case "infer":
      const result = MnistModel.infer(data);
      self.postMessage({ type: "infer", data: result });
      break;
    case "train":
      const loss = MnistModel.train(data.batch, data.learningRate);
      self.postMessage({ type: "train", data: loss });
      break;
  }
});
