import { defineConfig, type Plugin } from "vite";
import { readFileSync, existsSync, mkdirSync, copyFileSync, readdirSync } from "node:fs";
import { resolve, join } from "node:path";

const ROOT = resolve(import.meta.dirname);

/** Serves data/ and reports/ in dev and copies them into dist on build. */
function artifacts(): Plugin {
  const dirs = ["data", "reports"];
  return {
    name: "artifact-files",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        for (const dir of dirs) {
          if (url.startsWith(`/${dir}/`)) {
            const file = resolve(ROOT, url.slice(1));
            if (file.startsWith(resolve(ROOT, dir)) && existsSync(file)) {
              res.setHeader("Content-Type", "application/json; charset=utf-8");
              res.end(readFileSync(file));
              return;
            }
          }
        }
        next();
      });
    },
    closeBundle() {
      const out = join(ROOT, "dist");
      for (const dir of dirs) {
        const src = join(ROOT, dir);
        if (!existsSync(src)) continue;
        const dst = join(out, dir);
        mkdirSync(dst, { recursive: true });
        for (const f of readdirSync(src)) {
          if (f.endsWith(".json")) copyFileSync(join(src, f), join(dst, f));
        }
      }
    },
  };
}

export default defineConfig({
  root: ROOT,
  plugins: [artifacts()],
  server: {
    port: 5173,
    strictPort: true,
    fs: { allow: [ROOT] },
  },
  publicDir: "public",
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
  },
  optimizeDeps: { include: ["three"] },
});
