class MnistCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.shadowRoot.innerHTML = `
      <style>
        :host {
          display: block;
          width: 100px;
          height: 100px;
          border: 1px solid black;
          background-color: white;
          font-size: 20px;
          text-align: center;
        }
      </style>
      <slot></slot>
    `;
  }
}
