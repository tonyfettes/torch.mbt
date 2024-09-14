class MnistImage extends HTMLCanvasElement {
  constructor() {
    super();
    this.classList.add("mnist-image");
  }
  connectedCallback() {
    this.width = 28;
    this.height = 28;
    const jsonUrl = this.attributes.getNamedItem("src")?.value;
    if (!jsonUrl) {
      return;
    }
    const context = this.getContext("2d");
    fetch(jsonUrl)
      .then((response) => response.json())
      .then((json) => {
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
      });
  }
}

customElements.define("mnist-image", MnistImage, { extends: "canvas" });

export default MnistImage;
