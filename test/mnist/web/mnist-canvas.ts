import WebComponent from "./web-component";
import mnistWorker from "./mnist-worker";
import template from "./mnist-canvas.html?template";

class MnistCanvas extends WebComponent(HTMLElement) {
  private isDrawing: boolean;
  private lastX: number;
  private lastY: number;
  constructor() {
    super();
    this.isDrawing = false;
    this.lastX = 0;
    this.lastY = 0;
  }
  setImageData(imageData: ImageData) {
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
  private dispatchDrawEvent(imageData: ImageData): void {
    const data = imageData.data;
    const input = new Float32Array(data.length / 4);
    for (let i = 0; i < data.length; i += 4) {
      input[i / 4] = data[i + 3] / 255.0;
    }
    mnistWorker.postMessage({ type: "infer", data: input });
  }
  override connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot?.appendChild(template.content.cloneNode(true));
    const canvas = this.shadowRoot?.querySelector("canvas");
    if (!canvas) {
      return;
    }
    canvas.style.width = "224px";
    canvas.style.height = "224px";
    const context = canvas.getContext("2d", { willReadFrequently: true });
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

export default MnistCanvas;
