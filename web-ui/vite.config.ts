import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev proxy forwards the browser's address in X-Forwarded-For (xfwd) and connects to
// 127.0.0.1, the only address uvicorn trusts that header from. The trace API
// (capx_trace_viewer/trace_api.py) answers loopback clients only, so with `npm run dev -- --host`
// a LAN client still gets 403 from /api/traces instead of looking like a local one.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8290',
        changeOrigin: true,
        xfwd: true,
      },
    },
  },
})
