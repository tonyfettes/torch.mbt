import mnistWorker from "./mnist-worker";
import WebComponent from "./web-component";
import template from "./mnist-chart.html?template";

class MnistChart extends WebComponent(HTMLElement) {
  loadChart(data: number[]): void {
    const wrapper = this.shadowRoot?.querySelector("div#wrapper");
    if (!wrapper) {
      return;
    }
    for (const element of wrapper.children) {
      wrapper.removeChild(element);
    }
    const chart = document.createElement("div");
    chart.style.display = "flex";
    chart.style.flexDirection = "row";
    chart.style.alignItems = "end";
    for (let i = 0; i <= 10; i = i + 1) {
      const column = document.createElement("div");
      column.style.width = "28px";
      const bar = document.createElement("div");
      bar.style.width = "28px";
      bar.style.height = `${(28 * 8 * data[i]) >>> 0}px`;
      bar.style.backgroundColor = "rgba(0, 0, 0, 1.0)";
      bar.style.display = "flex";
      bar.style.flexDirection = "column";
      bar.style.alignItems = "center";
      const label = document.createElement("div");
      label.innerText = i.toString();
      column.appendChild(bar);
      column.appendChild(label);
      chart.appendChild(column);
    }
    wrapper.appendChild(chart);
  }
  connectedCallback() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot?.appendChild(template.content.cloneNode(true));
    this.loadChart(Array(10).fill(0.0));
    mnistWorker.addEventListener("message", (event) => {
      const { type, data } = event.data;
      if (type === "infer") {
        const max = Math.max(...data);
        const exp = [];
        for (const val of data) {
          exp.push(Math.exp(val - max));
        }
        const sum = exp.reduce((a, b) => a + b, 0);
        const probabilities = [];
        for (const value of exp) {
          probabilities.push(value / sum);
        }
        this.loadChart(probabilities);
      }
    });
  }
}

export default MnistChart;
