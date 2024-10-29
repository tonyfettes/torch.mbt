import MnistCanvas from "./mnist-canvas";
import MnistCard from "./mnist-card";
import MnistChart from "./mnist-chart";
import MnistGallery from "./mnist-gallery";
import MnistImage from "./mnist-image";
import MnistTrainer from "./mnist-trainer";

customElements.define("mnist-card", MnistCard);
customElements.define("mnist-chart", MnistChart);
customElements.define("mnist-image", MnistImage);
customElements.define("mnist-gallery", MnistGallery);
customElements.define("mnist-canvas", MnistCanvas);
customElements.define("mnist-trainer", MnistTrainer);

declare global {
  interface HTMLElementTagNameMap {
    "mnist-card": MnistCard;
    "mnist-chart": MnistChart;
    "mnist-image": MnistImage;
    "mnist-gallery": MnistGallery;
    "mnist-canvas": MnistCanvas;
    "mnist-trainer": MnistTrainer;
  }
  interface HTMLElementEventMap {
    "mnist-select": CustomEvent<ImageData>;
    "mnist-draw": CustomEvent<Uint8Array>;
    "mnist-image-load": Event;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const mnistGalleries = document.querySelectorAll("mnist-gallery");
  const mnistCanvas = document.querySelector("mnist-canvas");
  mnistGalleries.forEach((gallery) => {
    gallery?.addEventListener("mnist-select", (event) => {
      mnistCanvas?.setImageData(event.detail);
    });
  });
});
