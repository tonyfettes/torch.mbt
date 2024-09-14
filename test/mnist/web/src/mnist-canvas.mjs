class MnistCanvas extends HTMLCanvasElement {
  constructor() {
    super();
    this.classList.add("mnist-canvas");
  }
  connectedCallback() {
    this.style.width = "224px";
    this.style.height = "224px";
    const context = this.getContext("2d");
    if (!context) {
      return;
    }
    this.addEventListener("mousedown", (event) => {
      this.isDrawing = true;
      this.lastX = (event.offsetX / 8) >>> 0;
      this.lastY = (event.offsetY / 8) >>> 0;
      context.beginPath();
    });
    this.addEventListener("mousemove", (event) => {
      if (this.isDrawing) {
        context.moveTo(this.lastX, this.lastY);
        const x = (event.offsetX / 8) >>> 0;
        const y = (event.offsetY / 8) >>> 0;
        context.lineTo(x, y);
        this.lastX = x;
        this.lastY = y;
        context.stroke();
        const imageData = context.getImageData(0, 0, 28, 28);
        const data = imageData.data;
        const input = new Float64Array(data.length / 4);
        for (let i = 0; i < data.length; i += 4) {
          input[i / 4] = data[i + 3] / 255;
        }
        this.dispatchEvent(
          new CustomEvent("draw", {
            detail: input,
          })
        );
      }
    });
    this.addEventListener("mouseup", () => {
      this.isDrawing = false;
      context.closePath();
    });
  }
}

customElements.define("mnist-canvas", MnistCanvas, { extends: "canvas" });

export default MnistCanvas;
