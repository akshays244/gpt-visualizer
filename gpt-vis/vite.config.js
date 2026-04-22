import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: "/gpt-visualizer/",   // 👈 ADD THIS LINE
  plugins: [react()],
})