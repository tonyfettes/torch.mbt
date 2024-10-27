import * as MnistModel from "./mnist-model";

export type MnistWorkerRequest =
  | {
      type: "infer";
      data: MnistModel.Input;
    }
  | {
      type: "train";
      data: MnistModel.Batch;
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
      const loss = MnistModel.train(data, 0.001);
      self.postMessage({ type: "train", data: loss });
      break;
  }
});
