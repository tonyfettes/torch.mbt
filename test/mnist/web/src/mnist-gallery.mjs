// @ts-check
import MnistCard from "./mnist-card.mjs";
import mnistDatabase from "./mnist-database.mjs";
import WebComponent from "./web-component.mjs";

/**
 * MNIST Gallery
 */
class MnistGallery extends WebComponent(HTMLElement) {
  /** @override */
  static observedAttributes = ["src"];
  /**
   * @param {string} name
   * @param {string} oldValue
   * @param {string} newValue
   * @override
   */
  attributeChangedCallback(name, oldValue, newValue) {
    if (name === "src") {
      for (const element of this.shadowRoot?.children ?? []) {
        this.shadowRoot?.removeChild(element);
      }
      this.loadGallery(newValue);
    }
  }
  /**
   * @param {string} src
   * @returns {Promise<void>}
   */
  async loadGallery(src) {
    const shadowRoot = this.shadowRoot;
    if (!shadowRoot) {
      return;
    }
    // const column = (document.body.clientWidth / 32) >>> 0;
    const column = 32;
    const row = 4;
    for (let i = 0; i < row; i = i + 1) {
      const rowDiv = document.createElement("div");
      rowDiv.style.display = "flex";
      rowDiv.style.flexDirection = "row";
      rowDiv.style.alignItems = "center";
      rowDiv.style.justifyContent = "center";
      rowDiv.style.gap = "8px";
      for (let j = 0; j < column; j = j + 1) {
        const card = document.createElement("mnist-card");
        mnistDatabase.getData(src, i * column + j).then((data) => {
          const image = new Uint8Array(data.image);
          const imageBlob = new Blob([image]);
          const imageBlobURL = URL.createObjectURL(imageBlob);
          card.setAttribute("image", imageBlobURL);
          card.addEventListener("mnist-image-load", () => {
            URL.revokeObjectURL(imageBlobURL);
          });
          card.addEventListener("mnist-select", (event) => {
            this.dispatchEvent(
              new CustomEvent("mnist-select", {
                detail: event.detail,
              }),
            );
          });
          card.setAttribute("label", data.label.toString());
        });
        rowDiv.appendChild(card);
      }
      this.shadowRoot.appendChild(rowDiv);
    }
  }
  /**
   * @override
   */
  connectedCallback() {
    this.style.display = "flex";
    this.style.flexDirection = "column";
    this.style.alignItems = "center";
    this.style.justifyContent = "center";
    this.style.gap = "8px";
    // this.style.overflow = "scroll";
    this.attachShadow({ mode: "open" });
    const src = this.getAttribute("src");
    if (src) {
      this.loadGallery(src);
    }
  }
}

customElements.define("mnist-gallery", MnistGallery);

export default MnistGallery;
