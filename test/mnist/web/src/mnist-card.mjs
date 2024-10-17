import WebComponent from "./web-component.mjs";
import "./mnist-image.mjs";

// @ts-check
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

/**
 * MNIST Card.
 */
class MnistCard extends WebComponent(HTMLElement) {
  /**
   * @override
   */
  static observedAttributes = ["image", "label"];
  /**
   * Load the image.
   * @param {string?} url - The URL of the image
   * @returns {Promise<void>}
   * @private
   */
  async loadImage(url) {
    const onClick = () => {
      const context = this._image?.getContext();
      if (!context) {
        return;
      }
      this.dispatchEvent(
        new CustomEvent("mnist-select", {
          detail: context.getImageData(0, 0, 28, 28),
        }),
      );
    };
    if (!this._image) {
      const image = document.createElement("mnist-image");
      this.shadowRoot?.querySelector("#wrapper")?.appendChild(image);
      this._image = image;
    }
    if (url) {
      this._image.setAttribute("src", url);
      this._image.addEventListener("mnist-image-load", () => {
        this.addEventListener("click", onClick);
      });
    } else {
      this.removeEventListener("click", onClick);
      this._image.removeAttribute("src");
    }
  }
  /**
   * @param {string} label
   */
  async createLabel(label) {
    const template = labelTemplate.content.cloneNode(true);
    const shadowRoot = this.shadowRoot;
    if (!shadowRoot) {
      return;
    }
    shadowRoot.querySelector("#wrapper")?.appendChild(template);
    const labelDiv = shadowRoot.querySelector(".mnist-label");
    if (!labelDiv) {
      return;
    }
    labelDiv.innerHTML = label;
  }
  /**
   * @param {string} label
   */
  async loadLabel(label) {
    const labelElement = this.shadowRoot?.querySelector(".mnist-label");
    if (!labelElement) {
      return await this.createLabel(label);
    }
    labelElement.innerHTML = label;
  }
  /**
   * @param {string} name
   * @param {any} _oldValue
   * @param {any} newValue
   * @override
   */
  attributeChangedCallback(name, _oldValue, newValue) {
    if (name === "image") {
      this.loadImage(newValue);
    } else if (name === "label") {
      this.loadLabel(newValue);
    }
  }
  connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot?.appendChild(template.content.cloneNode(true));
    this.loadImage(this.getAttribute("image"));
    const label = this.getAttribute("label");
    if (label) {
      this.loadLabel(label);
    }
  }
}

customElements.define("mnist-card", MnistCard);

export default MnistCard;
