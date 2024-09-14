import "./mnist-image.mjs";

console.log("HERE");

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

class MnistCard extends HTMLDivElement {
  constructor() {
    super();
    this.classList.add("mnist-card");
  }
  loadImage() {
    console.log("Loading image");
    /** @type {HTMLCanvasElement} */
    const image = document.createElement("canvas", {
      is: "mnist-image",
    });
    image.setAttribute("src", `/data/${this._split}_${this._id}.json`);
    image.addEventListener("load", () => {
      const template = labelTemplate.content.cloneNode(true);
      template.querySelector(".mnist-label").innerHTML =
        image.getAttribute("label");
      this.shadowRoot.querySelector("#wrapper").appendChild(template);
      this.addEventListener("mousedown", () => {
        console.log("Selecting image");
        const context = image.getContext("2d");
        this.dispatchEvent(
          new CustomEvent("select", {
            detail: context.getImageData(0, 0, 28, 28),
          })
        );
      });
    });
    if (!this._image) {
      this.shadowRoot.querySelector("#wrapper").appendChild(image);
    } else {
      this.shadowRoot
        .querySelector("#wrapper")
        .replaceChild(image, this._image);
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
    this.shadowRoot.appendChild(template.content.cloneNode(true));

    this._split = this.getAttribute("data-split");
    this._id = this.getAttribute("data-id");

    if (this._split && this._id) {
      this.loadImage();
    }
  }
}

customElements.define("mnist-card", MnistCard, { extends: "div" });

export default MnistCard;
