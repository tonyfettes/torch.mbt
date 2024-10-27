import type { MnistWorkerRequest, MnistWorkerResponse } from "./mnist.worker";

interface MnistWorker extends Worker {
  postMessage(message: MnistWorkerRequest): void;
  addEventListener(
    type: "message",
    listener: (this: MnistWorker, ev: MessageEvent<MnistWorkerResponse>) => any,
    options?: boolean | AddEventListenerOptions
  ): void;
  addEventListener(
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | AddEventListenerOptions
  ): void;
}

const mnistWorker: MnistWorker = new Worker(
  new URL("./mnist.worker.ts", import.meta.url),
  { type: "module" }
);

export default mnistWorker;
