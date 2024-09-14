import MnistCanvas from "./mnist-canvas.mjs";

const mnistWorker = new Worker("/mnist-worker.mjs", {
  type: "module",
});
const mnistGalleries = document.querySelectorAll("mnist-gallery");
/** @type {MnistCanvas?} */
const mnistCanvas = document.querySelector("mnist-canvas");
const mnistChart = document.querySelector("mnist-chart");
const mnistCanvasClearButton = document.querySelector(
  "#mnist-canvas-clear-button"
);
mnistCanvasClearButton?.addEventListener("click", () => {
  mnistCanvas?.setImageData(new ImageData(28, 28));
});
mnistGalleries.forEach((gallery) => {
  gallery?.addEventListener(
    "select",
    /** @param {CustomEvent<ImageData>} event */
    (event) => {
      mnistCanvas?.setImageData(event.detail);
    }
  );
});
mnistCanvas?.addEventListener(
  "draw",
  /** @param {CustomEvent<Float64Array>} event */
  (event) => {
    mnistWorker.postMessage({ type: "infer", data: event.detail });
  }
);
mnistWorker.addEventListener(
  "message",
  /** @param {MessageEvent<{ type: string, data: Float64Array }>} event */
  (event) => {
    const { type, data } = event.data;
    if (type === "infer") {
      const output = Array.from(data);
      mnistChart?.setAttribute("data", JSON.stringify(output));
    }
  }
);
