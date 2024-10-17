// @ts-check
import "./mnist-gallery.mjs";
import "./mnist-canvas.mjs";
import "./mnist-chart.mjs";
import "./mnist-card.mjs";
import "./mnist-image.mjs";
import "./mnist-trainer.mjs";

document.addEventListener("DOMContentLoaded", () => {
  const mnistGalleries = document.querySelectorAll("mnist-gallery");
  const mnistCanvas = document.querySelector("mnist-canvas");
  mnistGalleries.forEach((gallery) => {
    gallery?.addEventListener("mnist-select", (event) => {
      mnistCanvas?.setImageData(event.detail);
    });
  });
});
