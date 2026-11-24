import { test, expect } from "@playwright/test";

test.describe.configure({ mode: "serial" });

test("隐藏答案从不进入观测向量或动作选择", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  const result = await page.evaluate(async () => {
    const { generateScenario } = await import("/src/core/scenario.ts");
    const { EpisodeRunner } = await import("/src/core/engine.ts");
    const forbidden = ["hiddenTargetSlot", "hiddenTargetGlyph", "targetSlot"];
    let leaks = 0;
    for (let seed = 9300; seed < 9320; seed++) {
      const sc = generateScenario(seed);
      const runner = new EpisodeRunner(sc);
      while (!runner.ended) {
        const obs = runner.observation();
        const text = JSON.stringify({ v: obs.vector, s: obs.snapshot, l: obs.legalActions });
        for (const key of forbidden) if (text.includes(key)) leaks++;
        const legal = obs.legalActions.find((a) => a.legal);
        if (!legal) break;
        runner.step(legal.action);
      }
    }
    return { leaks };
  });
  expect(result.leaks).toBe(0);
});

test("训练与评估种子互不重叠", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const c = await import("/src/core/constants.ts");
    const ranges: Array<[number, number]> = [
      [c.TRAIN_SEED_START, c.TRAIN_SEED_END],
      [c.TRAIN_REFUSAL_START, c.TRAIN_REFUSAL_END],
      [c.DEV_SEED_START, c.DEV_SEED_END],
      [c.VAL_SEED_START, c.VAL_SEED_END],
      [c.VAL_REFUSAL_START, c.VAL_REFUSAL_END],
      [c.HIDDEN_STANDARD_START, c.HIDDEN_STANDARD_END],
      [c.HIDDEN_AMBIGUOUS_START, c.HIDDEN_AMBIGUOUS_END],
      [c.HIDDEN_CLUTTERED_START, c.HIDDEN_CLUTTERED_END],
      [c.HIDDEN_UNSOLVABLE_START, c.HIDDEN_UNSOLVABLE_END],
    ];
    for (let i = 0; i < ranges.length; i++) {
      for (let j = i + 1; j < ranges.length; j++) {
        const [a0, a1] = ranges[i];
        const [b0, b1] = ranges[j];
        if (a0 <= b1 && b0 <= a1) return { ok: false, conflict: [ranges[i], ranges[j]] };
      }
    }
    return { ok: true };
  });
  expect(result.ok).toBe(true);
});

test("示范轨迹重放能精确复现记录的完整回合", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const { generateScenario } = await import("/src/core/scenario.ts");
    const { runEpisode } = await import("/src/core/engine.ts");
    const { makeReplayPolicy } = await import("/src/core/policies.ts");
    const res = await fetch("/data/train-demonstrations.json", { cache: "no-store" });
    const dataset = await res.json();
    const bySeed = new Map(dataset.episodes.map((e: { seed: number; actions: number[] }) => [e.seed, e.actions]));
    let checked = 0;
    let mismatch = 0;
    for (const ep of dataset.episodes as Array<{ seed: number; actions: number[]; endReason: string | null }>) {
      if (ep.actions.length === 0) continue;
      const replay = runEpisode(generateScenario(ep.seed), makeReplayPolicy("replay", bySeed as Map<number, number[]>));
      checked++;
      if (replay.endReason !== ep.endReason) mismatch++;
    }
    return { checked, mismatch };
  });
  expect(result.checked).toBeGreaterThan(300);
  expect(result.mismatch).toBe(0);
});

test("所有可解种子都可由完整回合完成，且合法性约束生效", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const { generateScenario } = await import("/src/core/scenario.ts");
    const { EpisodeRunner } = await import("/src/core/engine.ts");
    const { makeRuleDemonstrator } = await import("/src/core/policies.ts");
    let checked = 0;
    let ok = 0;
    let illegalAccepted = 0;
    for (let seed = 9300; seed < 9340; seed++) {
      const sc = generateScenario(seed);
      const runner = new EpisodeRunner(sc);
      const policy = makeRuleDemonstrator();
      while (!runner.ended) {
        const obs = runner.observation();
        const action = policy.select(obs);
        const entry = obs.legalActions[action];
        if (!entry?.legal) illegalAccepted++;
        const rec = runner.step(action, policy);
        if (!rec) break;
      }
      checked++;
      if (runner.endReason === "success") ok++;
    }
    return { checked, ok, illegalAccepted };
  });
  expect(result.illegalAccepted).toBe(0);
  expect(result.ok).toBe(result.checked);
});

test("仲裁代理只发出合法动作，且只在新布局上重算通过率", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#model-chip")).toContainText("模型", { timeout: 15000 });
  const result = await page.evaluate(async () => {
    const { generateScenario } = await import("/src/core/scenario.ts");
    const { runEpisode } = await import("/src/core/engine.ts");
    const { makeArbiter } = await import("/src/core/arbiter.ts");
    const { MLP } = await import("/src/core/model.ts");
    const linearRes = await fetch("/models/experiment-linear.json", { cache: "no-store" });
    const mlpRes = await fetch("/models/experiment-mlp.json", { cache: "no-store" });
    const cfgRes = await fetch("/models/arbiter-config.json", { cache: "no-store" });
    const linear = MLP.fromJSON(await linearRes.json());
    const mlp = MLP.fromJSON(await mlpRes.json());
    const cfg = await cfgRes.json();
    const out: Record<string, { pass: number; requests: number; illegal: number; wrong: number; forward: number }> = {};
    for (const [kind, net] of [["linear", linear], ["mlp", mlp]] as const) {
      for (const mode of ["defer", "gather", "both"] as const) {
        const policy = () =>
          makeArbiter(net, { modelKind: kind, modelVersion: "test", config: { threshold: 0.8, mode, maxDeferrals: null } });
        let pass = 0;
        let requests = 0;
        let illegal = 0;
        let wrong = 0;
        let forward = 0;
        for (let seed = 3000; seed <= 3049; seed++) {
          const res = runEpisode(generateScenario(seed), policy());
          if (res.success || res.safeRefusal) pass++;
          requests += res.demonstrationRequests;
          wrong += res.endReason === "wrong-target" ? 1 : 0;
          forward += res.inference.forwardPasses;
          for (const step of res.steps) if (!step.legal) illegal++;
        }
        out[`${kind}:${mode}`] = { pass, requests, illegal, wrong, forward };
      }
    }
    return { out, trigger: cfg.trigger };
  });
  for (const [key, row] of Object.entries(result.out)) {
    expect(row.illegal, `${key} 出现非法动作`).toBe(0);
    expect(row.forward, `${key} 前向次数应为正`).toBeGreaterThan(0);
    expect(row.pass, `${key} 新布局通过数应远高于随机`).toBeGreaterThanOrEqual(40);
  }
  expect(result.trigger.threshold).toBeGreaterThan(0);
  expect(["defer", "gather", "both"]).toContain(result.trigger.mode);
});

test("仲裁请求次数与持久化触发条件一致", async ({ page }) => {
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const { generateScenario } = await import("/src/core/scenario.ts");
    const { runEpisode } = await import("/src/core/engine.ts");
    const { makeArbiter } = await import("/src/core/arbiter.ts");
    const { MLP } = await import("/src/core/model.ts");
    const cfg = await (await fetch("/models/arbiter-config.json", { cache: "no-store" })).json();
    const net = MLP.fromJSON(await (await fetch("/models/experiment-linear.json", { cache: "no-store" })).json());
    const policy = () =>
      makeArbiter(net, {
        modelKind: "linear",
        modelVersion: "test",
        config: { threshold: cfg.trigger.threshold, mode: cfg.trigger.mode, maxDeferrals: null },
      });
    // High threshold must never produce fewer requests than the persisted one.
    let persistedRequests = 0;
    let highRequests = 0;
    for (let seed = 3000; seed <= 3019; seed++) {
      persistedRequests += runEpisode(generateScenario(seed), policy()).demonstrationRequests;
      const high = makeArbiter(net, {
        modelKind: "linear",
        modelVersion: "test",
        config: { threshold: Math.min(0.99, cfg.trigger.threshold + 0.15), mode: cfg.trigger.mode, maxDeferrals: null },
      });
      highRequests += runEpisode(generateScenario(seed), high).demonstrationRequests;
    }
    return { persistedRequests, highRequests, threshold: cfg.trigger.threshold };
  });
  expect(result.threshold).toBeGreaterThan(0);
  expect(result.highRequests).toBeGreaterThanOrEqual(result.persistedRequests);
});
