/** @type {WebAssembly.Imports} */
const importObject = {
  spectest: {
    print_char: (() => {
      /** @type {number[]} buffer */
      let buffer = [];
      function flush() {
        if (buffer.length > 0) {
          console.log(
            new TextDecoder("utf-16").decode(new Uint16Array(buffer).valueOf())
          );
          buffer = [];
        }
      }
      /** @param {number} ch */
      function log(ch) {
        if (ch == "\n".charCodeAt(0)) {
          flush();
        } else if (ch == "\r".charCodeAt(0)) {
          /* noop */
        } else {
          buffer.push(ch);
        }
      }
      return log;
    })(),
  }
}

WebAssembly.instantiateStreaming(fetch("/worker.wasm"), importObject).then(
)
