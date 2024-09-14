// @ts-check
import MnistImage from "./mnist-image.mjs";

const template = document.createElement("template");
template.innerHTML = `<div
  id="wrapper"
  style="display: flex; flex-direction: column; align-items: center; position: relative; border: 1px solid black;"
></div>`;

const labelTemplate = document.createElement("template");
labelTemplate.innerHTML = `<style>
  .mnist-label {
    position: absolute;
    top: 0px;
    left: 0px;
    width: 28px;
    height: 28px;
    opacity: 0.0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background-color: white;
    transition: opacity 0.2s;
  }
  .mnist-label:hover {
    opacity: 1.0;
  }
</style>
<div class="mnist-label"></div>`;

class MnistCard extends HTMLElement {
  constructor() {
    super();
  }
  loadImage() {
    const image = document.createElement("mnist-image");
    if (!(image instanceof MnistImage)) {
      return;
    }
    image.setAttribute("src", `/data/${this._split}_${this._id}.json`);
    image.addEventListener("load", () => {
      const template = labelTemplate.content.cloneNode(true);
      const shadowRoot = this.shadowRoot;
      if (!shadowRoot) {
        return;
      }
      shadowRoot.querySelector("#wrapper")?.appendChild(template);
      const label = shadowRoot.querySelector(".mnist-label");
      if (!label) {
        return;
      }
      label.innerHTML = image.getAttribute("label") ?? "";
      const context = image.getContext();
      if (!context) {
        return;
      }
      this.addEventListener("mousedown", () => {
        this.dispatchEvent(
          new CustomEvent("select", {
            detail: context.getImageData(0, 0, 28, 28),
          })
        );
      });
    });
    if (!this._image) {
      this.shadowRoot?.querySelector("#wrapper")?.appendChild(image);
    } else {
      this.shadowRoot
        ?.querySelector("#wrapper")
        ?.replaceChild(image, this._image);
    }
    this._image = image;
  }
  attributeChangedCallback(name, _oldValue, newValue) {
    let updated = false;
    if (name === "data-id") {
      this._id = newValue;
      updated = true;
    }
    if (name === "data-split") {
      this._split = newValue;
      updated = true;
    }
    if (updated && this._id && this._split) {
      this.loadImage();
    }
  }
  connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot?.appendChild(template.content.cloneNode(true));

    this._split = this.getAttribute("data-split");
    this._id = this.getAttribute("data-id");

    if (this._split && this._id) {
      this.loadImage();
    }
  }
}

customElements.define("mnist-card", MnistCard);

export default MnistCard;
