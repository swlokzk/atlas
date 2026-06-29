import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import mdx from "@mdx-js/rollup";

export default defineConfig({
  plugins: [
    { enforce: "pre", ...mdx() },
    react({ include: /\.(mdx|js|jsx|ts|tsx)$/ }),
  ],
  assetsInclude: ["**/*.lottie"], // Ensure .lottie files are treated as assets
  build: {
    // sourcemap: true, // 禁用 Source Map 生成
    // 針對 3D 大體積專案，將警告閾值提高到 2000kB (2MB)
    chunkSizeWarningLimit: 2000,
  },
  // 設定 Rolldown / Rollup 的打包策略進行代碼分割
  rollupOptions: {
    output: {
      manualChunks(id) {
        // 將 node_modules 中的大型 3D 庫單獨打包成一個 vendor 檔案
        if (id.includes('node_modules')) {
          if (id.includes('three') || id.includes('@react-three') || id.includes('framer-motion')) {
            return 'vendor-3d';
          }
          return 'vendor'; // 其他第三方套件
        }
      }
    }
  }
});

