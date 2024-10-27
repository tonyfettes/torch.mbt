import { defineConfig, Plugin } from "vite";
import cp from "node:child_process";

function html(): Plugin {
  return {
    name: "html",

    watchChange(id) {
      if (id.endsWith(".mbt")) {
        cp.spawnSync("make", { stdio: "inherit" });
      }
    },

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
