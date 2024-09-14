// @ts-check

const template = document.createElement("template");
template.innerHTML = `<canvas
  width="28"
  height="28"
></canvas>`;

class MnistImage extends HTMLElement {
  observedAttributes = ["src"];
  constructor() {
    super();
  }
  /** @param {string} jsonUrl */
  async loadImage(jsonUrl) {
    const canvas = this.shadowRoot?.querySelector("canvas");
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const response = await fetch(jsonUrl);
    if (!response.ok) {
      return;
    }
    const json = await response.json();
    const image = json["image"];
    this.setAttribute("label", json["label"]);
    const imageData = context.getImageData(0, 0, 28, 28);
    for (let i = 0; i < 28; i = i + 1) {
      for (let j = 0; j < 28; j = j + 1) {
        imageData.data[(i * 28 + j) * 4 + 3] = image[i * 28 + j];
      }
    }
    context.putImageData(imageData, 0, 0);
    this.dispatchEvent(new Event("load"));
  }
  connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.style.height = "28px";
    this.style.width = "28px";
    this.shadowRoot?.appendChild(template.content.cloneNode(true));
    const jsonUrl = this.attributes.getNamedItem("src")?.value;
    if (!jsonUrl) {
      return;
    }
    this.loadImage(jsonUrl);
  }
  /**
   * @param {string} name
   * @param {any} _oldValue
   * @param {any} newValue
   */
  attributeChangedCallback(name, _oldValue, newValue) {
    if (name === "src") {
      this.loadImage(newValue);
    }
  }
  getContext() {
    const canvas = this.shadowRoot?.querySelector("canvas");
    if (!canvas) {
      return null;
    }
    return canvas.getContext("2d");
  }
}

customElements.define("mnist-image", MnistImage);

export default MnistImage;
