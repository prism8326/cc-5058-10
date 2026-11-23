import { defineConfig, devices } from "@playwright/test";

const LIB_PATH = [
  "/tmp/pwlibs/root/usr/lib/aarch64-linux-gnu",
  "/tmp/pwlibs/root/lib/aarch64-linux-gnu",
  process.env.LD_LIBRARY_PATH ?? "",
]
  .filter(Boolean)
  .join(":");

export default defineConfig({
  testDir: "tests",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          args: ["--use-gl=swiftshader", "--enable-unsafe-swiftshader"],
          env: { ...process.env, LD_LIBRARY_PATH: LIB_PATH },
        },
      },
    },
  ],
  webServer: {
    command: "npx vite --host 127.0.0.1 --port 5173",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
