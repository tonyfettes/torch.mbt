/**
 * Web Component mixin.
 */
function WebComponent<
  T extends {
    new (...args: any[]): HTMLElement;
    prototype: HTMLElement;
  },
>(Super: T) {
  abstract class WebComponent extends Super {
    static observedAttributes: string[];
    connectedCallback(): void {}
    attributeChangedCallback(
      _name: string,
      _oldValue: string | null,
      _newValue: string | null,
    ): void {}
  }
  return WebComponent;
}

export default WebComponent;
