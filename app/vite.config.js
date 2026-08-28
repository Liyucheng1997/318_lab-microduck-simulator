import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // MuJoCo WASM and onnxruntime-web stay on the CDN (dynamic import with
  // @vite-ignore in game/boot.js), exactly like the pre-Vite app: their
  // .wasm sidecars resolve relative to the CDN URL and never touch the
  // bundle.
  server: {
    port: 5173,
  },
  build: {
    // Keep the JS/CSS bundle out of dist/assets/: the game's static assets
    // (public/assets/) land there and must keep their historical URLs.
    assetsDir: "bundle",
    chunkSizeWarningLimit: 1500,
  },
});
