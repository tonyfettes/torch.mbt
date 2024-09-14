class MnistImage extends HTMLCanvasElement {
  constructor() {
    super();
    const jsonUrl = this.attributes.getNamedItem("src")?.value;
    if (!jsonUrl) {
      return;
    }
    const context = this.getContext("2d");
    fetch(jsonUrl)
      .then((response) => response.json())
      .then((json) => {
        const image = json["image"];
        const imageData = context.getImageData(0, 0, 28, 28);
        for (let i = 0; i < 28; i = i + 1) {
          for (let j = 0; j < 28; j = j + 1) {
            imageData.data[(i * 28 + j) * 4 + 3] = image[i * 28 + j];
          }
        }
        context.putImageData(imageData, 0, 0);
      });
  }
}

customElements.define("mnist-image", MnistImage, { extends: "canvas" });
