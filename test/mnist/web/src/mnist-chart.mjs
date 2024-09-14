// @ts-check

const template = document.createElement("template");
template.innerHTML = `<div
  id="wrapper"
  style="display: flex; flex-direction: row; align-items: end; height: 252px; width: fit-content;"
></div>`;

class MnistChart extends HTMLElement {
  static observedAttributes = ["data"];
  constructor() {
    super();
    /**
     * @private
     * @type {number[]}
     */
    this._data = Array(11).fill(0);
  }
  /**
   * @param {string} name
   * @param {any} _oldValue
   * @param {any} newValue
   * @returns {void}
   */
  attributeChangedCallback(name, _oldValue, newValue) {
    if (name === "data") {
      this._data = JSON.parse(newValue);
      this.loadChart();
    }
  }
  loadChart() {
    if (!this._data) {
      return;
    }
    const wrapper = this.shadowRoot?.querySelector("#wrapper");
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
      bar.style.height = `${(28 * 8 * this._data[i]) >>> 0}px`;
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
    const data = this.getAttribute("data");
    if (data) {
      this._data = JSON.parse(data);
    }
    this.loadChart();
  }
}

customElements.define("mnist-chart", MnistChart);

export default MnistChart;
