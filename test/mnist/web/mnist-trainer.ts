import WebComponent from "./web-component";
import template from "./mnist-trainer.html?template";
import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
import mnistWorker from "./mnist-worker";
import mnistDatabase from "./mnist-database";

class MnistTrainer extends WebComponent(HTMLElement) {
  private _logs: { step: number; loss: number }[];
  private _lossPlot: (SVGSVGElement | HTMLElement) | null;
  // private _state: "idle" | "busy";
  private _button: HTMLButtonElement | null;
  private _batchSize: number;
  static override observedAttributes = ["batch-size", "learning-rate"];
  constructor() {
    super();
    this._logs = [];
    this._lossPlot = null;
    // this._state = "idle";
    this._button = null;
    this._batchSize = 1;
  }
  private async plotLoss() {
    const newPlot = Plot.plot({
      marks: [Plot.lineY(this._logs, { x: "step", y: "loss" })],
    });
    if (this._lossPlot) {
      this.shadowRoot
        ?.querySelector("#loss")
        ?.replaceChild(newPlot, this._lossPlot);
    } else {
      this.shadowRoot?.querySelector("#loss")?.appendChild(newPlot);
    }
    this._lossPlot = newPlot;
  }
  /**
   * Train the model.
   */
  async train(): Promise<void> {
    mnistWorker.addEventListener("message", (event) => {
      const { type, data } = event.data;
      if (type !== "train") {
        return;
      }
      const step = this._logs.length;
      this._logs.push({
        step,
        loss: data,
      });
      this.plotLoss();
    });
    for (let epoch = 0; epoch < 1; epoch++) {
      for (let i = 0; i < 60000; i += this._batchSize) {
        const batch: [Float64Array, number][] = [];
        for (let b = 0; b < this._batchSize; b++) {
          const data = await mnistDatabase.getData("train", i + b);
          const input = new Float64Array(data.image.map((x) => x / 255));
          batch.push([input, data.label]);
        }
        mnistWorker.postMessage({
          type: "train",
          data: {
            batch,
            learningRate: 0.001
          }
        });
      }
    }
    if (this._button) {
      this._button.disabled = false;
    }
  }
  override connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot?.appendChild(template.content.cloneNode(true));
    this.plotLoss();
    const button: HTMLButtonElement | null | undefined =
      this.shadowRoot?.querySelector("button#start");
    if (button) {
      button.addEventListener("click", async () => {
        button.disabled = true;
        // this._state = "busy";
        await this.train();
      });
      this._button = button;
    }
    const batchSizeInput: HTMLInputElement | null | undefined =
      this.shadowRoot?.querySelector("input#batch-size");
    if (batchSizeInput) {
      batchSizeInput.addEventListener("input", () => {
        this._batchSize = Number(batchSizeInput.value);
      });
    }
  }
}

export default MnistTrainer;
