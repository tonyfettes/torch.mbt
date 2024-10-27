import { defineConfig, Plugin } from "vite";

function html(): Plugin {
  return {
    name: "html",

    transform(source, id) {
      if (id.endsWith(".html?template")) {
        return {
          code: `const template = document.createElement("template");
template.innerHTML = \`${source.trim()}\`;

export default template;
`,
          map: null,
        };
      }
    },
  };
}

export default defineConfig({
  plugins: [html()],
  build: {
    target: "esnext",
  },
  worker: {
    format: "es",
  },
});
