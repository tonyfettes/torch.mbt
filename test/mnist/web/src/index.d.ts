import type MnistCanvas from "./mnist-canvas.mjs";
import type MnistCard from "./mnist-card.mjs";
import type MnistGallery from "./mnist-gallery.mjs";
import type MnistImage from "./mnist-image.mjs";
import type MnistTrainer from "./mnist-trainer.mjs";

declare global {
  const Plot: typeof import("@observablehq/plot");

  interface HTMLElementTagNameMap {
    "mnist-card": MnistCard;
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
