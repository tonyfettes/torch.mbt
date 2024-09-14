import "./mnist-card.mjs";

class MnistGallery extends HTMLDivElement {
  constructor() {
    super();
    this.classList.add("mnist-gallery");
  }
  attributeChangedCallback(name, _oldValue, newValue) {
    if (name === "data-split") {
      this._split = newValue;
      for (const element of this.shadowRoot.children) {
        this.shadowRoot.removeChild(element);
      }
      this.loadGallery();
    }
  }
  loadGallery() {
    const column = (document.body.clientWidth / 32) >>> 0;
    const row = 4;
    for (let i = 0; i < row; i = i + 1) {
      const rowDiv = document.createElement("div");
      rowDiv.style.display = "flex";
      rowDiv.style.flexDirection = "row";
      rowDiv.style.alignItems = "center";
      rowDiv.style.justifyContent = "center";
      rowDiv.style.gap = "8px";
      for (let j = 0; j < column; j = j + 1) {
        const card = document.createElement("div", { is: "mnist-card" });
        card.setAttribute("data-split", this._split);
        card.setAttribute("data-id", j + i * column);
        card.addEventListener("select", (event) => {
          this.dispatchEvent(
            new CustomEvent("select", { detail: event.detail })
          );
        });
        rowDiv.appendChild(card);
      }
      this.shadowRoot.appendChild(rowDiv);
    }
  }
  connectedCallback() {
    this._split = this.getAttribute("data-split");
    this.style.display = "flex";
    this.style.flexDirection = "column";
    this.style.alignItems = "center";
    this.style.justifyContent = "center";
    this.style.gap = "8px";
    this.style.overflow = "scroll";
    this.attachShadow({ mode: "open" });
    this.loadGallery();
  }
}

customElements.define("mnist-gallery", MnistGallery, { extends: "div" });
