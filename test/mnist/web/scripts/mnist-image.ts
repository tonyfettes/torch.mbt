import WebComponent from "./web-component";
import template from "./mnist-image.html?template";

/**
 * MNIST image.
 */
class MnistImage extends WebComponent(HTMLElement) {
  static override observedAttributes = ["src"];
  /**
   * Load the image.
   */
  async loadImage(url: string) {
    const canvas = this.shadowRoot?.querySelector("canvas");
    if (!canvas) {
      console.error("MnistImage.loadImage.canvas");
      return;
    }
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) {
      console.error("MnistImage.loadImage.context");
      return;
    }
    if (!url) {
      console.error("MnistImage.loadImage.url");
      context.clearRect(0, 0, 28, 28);
      return;
    }
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch image from ${url}`);
    }
    const blob = await response.blob();
    const image = new Uint8Array(await blob.arrayBuffer());
    const imageData = context.getImageData(0, 0, 28, 28);
    for (let i = 0; i < 28; i = i + 1) {
      for (let j = 0; j < 28; j = j + 1) {
        imageData.data[(i * 28 + j) * 4 + 3] = image[i * 28 + j];
      }
    }
    context.putImageData(imageData, 0, 0);
    this.dispatchEvent(new Event("mnist-image-load"));
  }
  override connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.style.height = "28px";
    this.style.width = "28px";
    this.shadowRoot?.appendChild(template.content.cloneNode(true));
    const src = this.getAttribute("src");
    if (src) {
      this.loadImage(src);
    }
  }
  override attributeChangedCallback(name: string, _oldValue: string, newValue: string) {
    if (name === "src") {
      this.loadImage(newValue);
    }
  }
  getContext() {
    return this.shadowRoot?.querySelector("canvas")?.getContext("2d");
  }
}

export default MnistImage;
