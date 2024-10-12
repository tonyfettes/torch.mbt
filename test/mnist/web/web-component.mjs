/**
 * @typedef {{
 *   connectedCallback(): void;
 *   attributeChangedCallback(name: string, oldValue: string, newValue: string): void;
 * }} IWebComponent
 */

/**
 * @template {{
 *   new(...args: any[]): HTMLElement;
 *   prototype: HTMLElement;
 * }} T
 * @param {T} Super
 */
function WebComponent(Super) {
  /**
   * @implements {IWebComponent}
   * @abstract
   */
  class WebComponent extends Super {
    /**
     * @type {string[]}
     */
    static observedAttributes;
    connectedCallback() {}
    /**
     * @param {string} _name
     * @param {unknown} _oldValue
     * @param {unknown} _newValue
     */
    attributeChangedCallback(_name, _oldValue, _newValue) {}
  }
  return WebComponent;
}

export default WebComponent;
