import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Built straight into the directory FastAPI serves, so `npm run build`
  // followed by `uvicorn` is the whole deploy. No copy step to forget.
  build: { outDir: "../app/api/static", emptyOutDir: true },
  server: {
    // During `npm run dev` the API runs separately on 8000; proxying keeps the
    // browser on one origin so there is no CORS config that only exists for
    // development.
    proxy: {
      "/ask": "http://localhost:8000",
      "/sync": "http://localhost:8000",
      "/report": "http://localhost:8000",
      "/proposals": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
