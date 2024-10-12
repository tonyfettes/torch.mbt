const mnistWorker = new Worker("./mnist.worker.mjs", { type: "module" });

export default mnistWorker;
