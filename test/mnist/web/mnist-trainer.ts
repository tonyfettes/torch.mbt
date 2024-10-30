import WebComponent from "./web-component";
import template from "./mnist-trainer.html?template";
import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
import mnistWorker from "./mnist-worker";
import mnistDatabase from "./mnist-database";

class MnistTrainer extends WebComponent(HTMLElement) {
  private _logs: { step: number; loss: number }[];
  private _lossPlot: (SVGSVGElement | HTMLElement) | null;
  private _button: HTMLButtonElement | null;
  private _batchSizeInput: HTMLInputElement | null;
  private _learningRateInput: HTMLInputElement | null;
  private _epochInput: HTMLInputElement | null;
  private _training: {
    state: "running" | "stopped";
    epoch: number;
    index: number;
  };
  static override observedAttributes = ["batch-size", "learning-rate", "epoch"];
  constructor() {
    super();
    this._logs = [];
    this._lossPlot = null;
    this._button = null;
    this._batchSizeInput = null;
    this._learningRateInput = null;
    this._epochInput = null;
    this._training = {
      state: "stopped",
      epoch: 0,
      index: 0,
    };
  }
  private async plotLoss() {
    const newPlot = Plot.plot({
      marks: [
        Plot.lineY(this._logs.slice(Math.max(0, this._logs.length - 1000)), {
          x: "step",
          y: "loss",
        }),
      ],
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
  private get _epoch(): number {
    return Number(this._epochInput?.value) ?? 1;
  }
  private get _batchSize(): number {
    return Number(this._batchSizeInput?.value) ?? 1;
  }
  private get _learningRate(): number {
    return Number(this._learningRateInput?.value) ?? 0.1;
  }
  /**
   * Train the model.
   */
  async train(): Promise<void> {
    this._training.state = "running";
    for (let epoch = this._training.epoch; epoch < this._epoch; epoch++) {
      for (let i = this._training.index; i < 60000; i += this._batchSize) {
        if (this._training.state === "stopped") {
          this._training.epoch = epoch;
          this._training.index = i;
          return;
        }
        const batch: [Float64Array, number][] = [];
        for (let b = i; b < Math.min(i + this._batchSize, 60000); b++) {
          const data = await mnistDatabase.getData("train", b);
          const input = new Float64Array(data.image.map((x) => x / 255));
          batch.push([input, data.label]);
        }
        mnistWorker.postMessage({
          type: "train",
          data: {
            batch,
            learningRate: this._learningRate,
          },
        });
        await new Promise<void>((resolve) => {
          mnistWorker.addEventListener(
            "message",
            (event) => {
              const { type, data } = event.data;
              if (type !== "train") {
                return;
              }
              this._logs.push({
                step: epoch * 60000 + i,
                loss: data,
              });
              this.plotLoss();
              resolve();
            },
            {
              once: true,
            }
          );
        });
      }
    }
  }
  override connectedCallback() {
    this.attachShadow({ mode: "open" });
    if (!this.shadowRoot) {
      throw new Error("Failed to attach shadow");
    }
    this.shadowRoot.appendChild(template.content.cloneNode(true));
    this.plotLoss();
    this._button = this.shadowRoot.querySelector("button#start");
    const onClick = async () => {
      if (!this._button) {
        return;
      }
      console.log('Change the button value to "Abort"');
      this._button.innerText = "Stop";
      this._button?.addEventListener(
        "click",
        async () => {
          this._training.state = "stopped";
          if (this._button) {
            this._button.innerText = "Waiting";
          }
        },
        {
          once: true,
        }
      );
      this._batchSizeInput?.setAttribute("disabled", "");
      this._learningRateInput?.setAttribute("disabled", "");
      this._epochInput?.setAttribute("disabled", "");
      await this.train();
      this._button.innerText = "Start";
      this._button.addEventListener("click", onClick, {
        once: true,
      });
      this._batchSizeInput?.removeAttribute("disabled");
      this._learningRateInput?.removeAttribute("disabled");
      this._epochInput?.removeAttribute("disabled");
    };
    this._button?.addEventListener("click", onClick, {
      once: true,
    });
    this._batchSizeInput = this.shadowRoot?.querySelector("input#batch-size");
    this._learningRateInput = this.shadowRoot?.querySelector(
      "input#learning-rate"
    );
    this._epochInput = this.shadowRoot?.querySelector("input#epoch");
  }
}

export default MnistTrainer;
