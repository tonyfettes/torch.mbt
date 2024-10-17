import mnistWorker from "./mnist-worker.mjs";

// @ts-check
const template = document.createElement("template");
template.innerHTML = `<style>
  .mnist-canvas {
    display: flex;
    flex-direction: column;
    width: fit-content;
    gap: 8px;
  }
  .mnist-canvas-toolbar {
    display: flex;
    flex-direction: row;
    justify-content: center;
  }
</style>
<div class="mnist-canvas">
  <canvas
    width="28"
    height="28"
    style="border: 1px solid black; image-rendering: pixelated"
  ></canvas>
  <div class="mnist-canvas-toolbar">
    <button id="clear">Clear</button>
  </div>
</div>`;

class MnistCanvas extends HTMLElement {
  constructor() {
    super();
    /**
     * @private
     * @type {boolean}
     */
    this.isDrawing = false;
    /**
     * @private
     * @type {number}
     */
    this.lastX = 0;
    /**
     * @private
     * @type {number}
     */
    this.lastY = 0;
  }
  /**
   * @param {ImageData} imageData
   */
  setImageData(imageData) {
    const canvas = this.shadowRoot?.querySelector("canvas");
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    context.putImageData(imageData, 0, 0);
    this.dispatchDrawEvent(imageData);
  }
  /**
   * @param {ImageData} imageData
   * @returns {void}
   */
  dispatchDrawEvent(imageData) {
    const data = imageData.data;
    const input = new Uint8Array(data.length / 4);
    for (let i = 0; i < data.length; i += 4) {
      input[i / 4] = data[i + 3];
    }
    mnistWorker.postMessage({ type: "infer", data: input });
  }
  connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot?.appendChild(template.content.cloneNode(true));
    const canvas = this.shadowRoot?.querySelector("canvas");
    if (!canvas) {
      return;
    }
    canvas.style.width = "224px";
    canvas.style.height = "224px";
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const clearButton = this.shadowRoot?.querySelector("button#clear");
    clearButton?.addEventListener("click", () => {
      context.clearRect(0, 0, 28, 28);
      const imageData = context.getImageData(0, 0, 28, 28);
      this.dispatchDrawEvent(imageData);
    });
    this.addEventListener("mousedown", (event) => {
      const offsetX = event.clientX - canvas.getBoundingClientRect().left;
      const offsetY = event.clientY - canvas.getBoundingClientRect().top;
      this.isDrawing = true;
      this.lastX = (offsetX / 8) >>> 0;
      this.lastY = (offsetY / 8) >>> 0;
    });
    this.addEventListener("mouseleave", () => {
      this.isDrawing = false;
    });
    this.addEventListener("mousemove", (event) => {
      if (this.isDrawing) {
        const offsetX = event.clientX - canvas.getBoundingClientRect().left;
        const offsetY = event.clientY - canvas.getBoundingClientRect().top;
        context.beginPath();
        context.moveTo(this.lastX, this.lastY);
        const x = (offsetX / 8) >>> 0;
        const y = (offsetY / 8) >>> 0;
        context.lineTo(x, y);
        context.lineWidth = 2;
        context.stroke();
        this.lastX = x;
        this.lastY = y;
        const imageData = context.getImageData(0, 0, 28, 28);
        this.dispatchDrawEvent(imageData);
      }
    });
    this.addEventListener("mouseup", () => {
      this.isDrawing = false;
    });
  }
}

customElements.define("mnist-canvas", MnistCanvas);

export default MnistCanvas;
