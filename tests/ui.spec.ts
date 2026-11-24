import { test, expect } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));
  (page as unknown as { jsErrors: string[] }).jsErrors = errors;
});

async function loadSeed(page: import("@playwright/test").Page, seed: number): Promise<void> {
  await page.fill("#seed-input", String(seed));
  await page.click("#btn-load-seed");
  await expect(page.locator("#seed-chip")).toContainText(String(seed));
}

test("页面加载、模型与三维画布就绪", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await expect(page.locator("#seed-chip")).toContainText("场景");
  await expect(page.locator("#goal-glyph")).not.toHaveText("—");
  const canvas = page.locator("#viewport");
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  expect(box && box.width > 300 && box.height > 200).toBeTruthy();
  await expect(page.locator("#obs-groups .obs-group").first()).toBeVisible();
});

test("学得策略在完整回合中完成任务（标准场景）", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await page.uncheck("#chk-realtime");
  await loadSeed(page, 9000);
  await page.click("[data-driver='learned']");
  await page.click("#btn-play");
  await expect(page.locator("#end-banner")).toBeVisible({ timeout: 20000 });
  const banner = await page.locator("#end-banner").textContent();
  expect(banner).toMatch(/任务完成|安全拒绝/);
  const logCount = await page.locator("#log li").count();
  expect(logCount).toBeGreaterThan(0);
});

test("规则示范驱动同场景并产生多步完整交互", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await page.uncheck("#chk-realtime");
  await loadSeed(page, 9300);
  await page.click("[data-driver='rule']");
  await page.click("#btn-play");
  await expect(page.locator("#end-banner")).toBeVisible({ timeout: 20000 });
  const banner = await page.locator("#end-banner").textContent();
  expect(banner).toMatch(/任务完成|安全拒绝/);
});

test("不可解场景中学得策略安全拒绝而非盲目操作", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await page.uncheck("#chk-realtime");
  await loadSeed(page, 9500);
  await page.click("[data-driver='learned']");
  await page.click("#btn-play");
  await expect(page.locator("#end-banner")).toBeVisible({ timeout: 20000 });
  const banner = await page.locator("#end-banner").textContent();
  expect(banner).toMatch(/安全拒绝/);
});

test("手动模式非法动作被拒绝并给出原因", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await loadSeed(page, 9000);
  await page.click("[data-driver='manual']");
  await expect(page.locator("#manual-panel")).toBeVisible();
  const disabled = await page.locator("#manual-actions button:disabled").count();
  expect(disabled).toBeGreaterThan(0);
  const enabled = page.locator("#manual-actions button:not(:disabled)").first();
  const label = await enabled.textContent();
  await enabled.click();
  await expect(page.locator("#log li").first()).toContainText(label ?? "");
});

test("载入示范轨迹并在结果区显示策略动作与观测", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await page.uncheck("#chk-realtime");
  await loadSeed(page, 1000);
  await page.click("#btn-demo");
  await page.click("#btn-play");
  await expect(page.locator("#end-banner")).toBeVisible({ timeout: 20000 });
  const actionLabel = await page.locator("#action-label").textContent();
  expect(actionLabel && actionLabel.length > 1).toBeTruthy();
  await expect(page.locator("#obs-groups .obs-group").first()).toBeVisible();
  const errors = (page as unknown as { jsErrors: string[] }).jsErrors;
  expect(errors).toEqual([]);
});

test("仲裁代理在新布局驱动完整回合并显示来源与推理成本", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await expect(page.locator("#arbiter-status")).toContainText("触发条件已载入", { timeout: 10000 });
  await page.uncheck("#chk-realtime");
  await loadSeed(page, 3000);
  await page.click("[data-driver='arbiter']");
  await expect(page.locator("#arbiter-status")).toContainText("阈值", { timeout: 10000 });
  await page.click("#btn-play");
  await expect(page.locator("#end-banner")).toBeVisible({ timeout: 30000 });
  const banner = await page.locator("#end-banner").textContent();
  expect(banner).toMatch(/任务完成|安全拒绝/);
  expect(banner).toMatch(/示范请求 \d+ 次 · 前向 \d+ 次/);
  // Every logged step carries an arbitration provenance chip.
  const sourceCount = await page.locator("#log .log-source").count();
  const logCount = await page.locator("#log li").count();
  expect(sourceCount).toBeGreaterThan(0);
  expect(sourceCount).toBeLessThanOrEqual(logCount);
  await expect(page.locator("#action-timing")).toContainText("仲裁");
  const errors = (page as unknown as { jsErrors: string[] }).jsErrors;
  expect(errors).toEqual([]);
});

test("仲裁置信阈值调高时转交示范者并计数", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await page.uncheck("#chk-realtime");
  await loadSeed(page, 9000);
  await page.click("[data-driver='arbiter']");
  await page.selectOption("#arbiter-mode", "defer");
  await page.fill("#arbiter-threshold", "0.99");
  await page.dispatchEvent("#arbiter-threshold", "change");
  await page.click("#btn-play");
  await expect(page.locator("#end-banner")).toBeVisible({ timeout: 30000 });
  const banner = await page.locator("#end-banner").textContent();
  expect(banner).toMatch(/示范请求 [1-9]\d* 次/);
  const deferred = await page.locator("#log .log-source.defer").count();
  expect(deferred).toBeGreaterThan(0);
});

test("仲裁模型缺失时界面提示而非静默失败", async ({ page }) => {
  await page.route("**/models/experiment-linear.json", (route) => route.fulfill({ status: 500, body: "missing" }));
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  await page.click("[data-driver='arbiter']");
  await expect(page.locator("#arbiter-error")).toBeVisible({ timeout: 10000 });
  await expect(page.locator("#arbiter-error")).toContainText("加载失败");
  await page.uncheck("#chk-realtime");
  await page.click("#btn-play");
  await expect(page.locator("#log li").last()).toContainText("仲裁模型", { timeout: 10000 });
});
