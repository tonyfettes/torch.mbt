import WebComponent from "./web-component";
import MnistImage from "./mnist-image";
import template from "./mnist-card.html?template";

/**
 * MNIST Card.
 */
class MnistCard extends WebComponent(HTMLElement) {
  private _image: MnistImage | null = null;
  private labelDiv: HTMLDivElement | null = null;
  /**
   * @override
   */
  static observedAttributes = ["image", "label"];
  /**
   * Load the image.
   */
  private async loadImage(url: string): Promise<void> {
    const onClick = () => {
      const context = this._image?.getContext();
      if (!context) {
        return;
      }
      this.dispatchEvent(
        new CustomEvent("mnist-select", {
          detail: context.getImageData(0, 0, 28, 28),
        })
      );
    };
    if (!this._image) {
      const image = document.createElement("mnist-image");
      this.shadowRoot?.querySelector("#wrapper")?.appendChild(image);
      this._image = image;
    }
    if (url) {
      this._image?.setAttribute("src", url);
      this._image?.addEventListener("mnist-image-load", () => {
        this.addEventListener("click", onClick);
      });
    } else {
      this.removeEventListener("click", onClick);
      this._image?.removeAttribute("src");
    }
  }
  async createLabel(label: string) {
    const shadowRoot = this.shadowRoot;
    if (!shadowRoot) {
      return;
    }
    const labelDiv = document.createElement("div");
    labelDiv.classList.add("label");
    if (!labelDiv) {
      return;
    }
    labelDiv.innerHTML = label;
  }
  async loadLabel(label: string) {
    if (!this.labelDiv) {
      const labelDiv = document.createElement("div");
      labelDiv.classList.add("label");
      if (!labelDiv) {
        return;
      }
      labelDiv.innerHTML = label;
      this.labelDiv = labelDiv;
      return;
    }
    this.labelDiv.innerHTML = label;
  }
  override attributeChangedCallback(
    name: string,
    _oldValue: string,
    newValue: string
  ) {
    if (name === "image") {
      this.loadImage(newValue);
    } else if (name === "label") {
      this.loadLabel(newValue);
    }
  }
  override connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot?.appendChild(template.content.cloneNode(true));
    const image = this.getAttribute("image");
    if (image) {
      this.loadImage(image);
    }
    const label = this.getAttribute("label");
    if (label) {
      this.loadLabel(label);
    }
  }
}

export default MnistCard;
