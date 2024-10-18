import mnistDatabase from "./mnist-database.mjs";
import mnistWorker from "./mnist-worker.mjs";
import WebComponent from "./web-component.mjs";
import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";

/**
 * Create a template.
 * @param {string} html - The HTML string
 */
function createTemplate(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  return template;
}

class MnistTrainer extends WebComponent(HTMLElement) {
  /**
   * @type {HTMLTemplateElement}
   * @private
   */
  static template = createTemplate(`<div>
    <div id="mnist-train-loss"></div>
    <style>
      .mnist-train-control {
        display: flex;
        flex-direction: column;
        width: fit-content;
        gap: 8px;
      }
    </style>
    <div class="mnist-train-control">
      <div>
        <label for="mnist-batch-size">Batch Size:</label>
        <input type="number" id="mnist-batch-size" min="1" max="10000"></input>
      </div>
      <button id="mnist-train">Train</button>
    </div>
  </div>`);
  /**
   * @override
   * @type {string[]}
   */
  static observedAttributes = ["batch-size", "learning-rate"];
  constructor() {
    super();
    /**
     * @type {{ step: number, loss: number }[]}
     * @private
     */
    this._logs = [];
    /**
     * @type {((SVGSVGElement | HTMLElement) & import("@observablehq/plot").Plot)?}
     */
    this._lossPlot = null;
    /**
     * @type {"idle" | "busy"}
     * @private
     */
    this._state = "idle";
    /**
     * @type {HTMLButtonElement | null}
     * @private
     */
    this._button = null;
    /**
     * @type {number}
     * @private
     */
    this._batchSize = 1;
  }
  /**
   * @private
   */
  async plotLoss() {
    const newPlot = Plot.plot({
      marks: [Plot.lineY(this._logs, { x: "step", y: "loss" })],
    });
    if (this._lossPlot) {
      this.shadowRoot
        ?.querySelector("#mnist-train-loss")
        ?.replaceChild(newPlot, this._lossPlot);
    } else {
      this.shadowRoot?.querySelector("#mnist-train-loss")?.appendChild(newPlot);
    }
    this._lossPlot = newPlot;
  }
  /**
   * Train the model.
   * @returns {Promise<void>}
   */
  async train() {
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
        const batch = [];
        for (let b = 0; b < this._batchSize; b++) {
          const data = await mnistDatabase.getData("train", i);
          const input = new Uint8Array(data.image);
          batch.push({ input, label: data.label });
        }
        mnistWorker.postMessage({
          type: "train",
          data: batch,
        });
      }
    }
    if (this._button) {
      this._button.disabled = false;
    }
  }
  /**
   * @override
   */
  connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot?.appendChild(MnistTrainer.template.content.cloneNode(true));
    this.plotLoss();
    const button = this.shadowRoot?.querySelector("button");
    if (button) {
      button.addEventListener("click", async () => {
        button.disabled = true;
        this._state = "busy";
        await this.train();
      });
      this._button = button;
    }
    /** @type {HTMLInputElement | null | undefined} */
    const batchSizeInput = this.shadowRoot?.querySelector(
      "input#mnist-batch-size",
    );
    if (batchSizeInput) {
      batchSizeInput.addEventListener("input", () => {
        this._batchSize = Number(batchSizeInput.value);
      });
    }
  }
}

customElements.define("mnist-trainer", MnistTrainer);

export default MnistTrainer;
