// @ts-check
class MnistChart extends HTMLDivElement {
  static observedAttributes = ["data"];
  constructor() {
    super();
    /**
     * @private
     * @type {number[]}
     */
    this._data = [];
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
    for (const element of this.shadowRoot?.children ?? []) {
      this.shadowRoot?.removeChild(element);
    }
    const chart = document.createElement("div");
    chart.style.display = "flex";
    chart.style.flexDirection = "row";
    for (let i = 0; i <= 10; i = i + 1) {
      const bar = document.createElement("div");
      bar.style.width = "28px";
      bar.style.height = `${28 * 8 * this._data[i] >>> 0}px`;
      bar.style.backgroundColor = "rgba(0, 0, 0, 0.1)";
      bar.style.display = "flex";
      bar.style.flexDirection = "column";
      bar.style.alignItems = "center";
      chart.appendChild(bar);
    }
  }
  connectedCallback() {
    this.attachShadow({ mode: "open" });
    const data = this.getAttribute("data");
    console.log("connectedCallback", data);
    if (data) {
      this._data = JSON.parse(data);
      this.loadChart();
    }
  }
}

customElements.define("mnist-chart", MnistChart, { extends: "div" });

export default MnistChart;
