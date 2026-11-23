import { writeFileSync, mkdirSync, readFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { runEpisode, EpisodeRunner } from "../src/core/engine.js";
import { ruleDemonstrator, makeRandomPolicy, makeLearnedPolicy, demonstratorAction } from "../src/core/policies.js";
import { generateScenario } from "../src/core/scenario.js";
import { MLP, type ModelJSON } from "../src/core/model.js";
import { validateModelFormat } from "../src/core/validate.js";
import { hiddenSeeds, validationSeeds, collectEpisode } from "./dataset.js";
import type { EpisodeResult, Policy, Scenario } from "../src/core/types.js";

const ROOT = resolve(import.meta.dirname, "..");
const MODEL_PATH = join(ROOT, "public", "models", "learned-model.json");

interface Aggregate {
  policy: string;
  group: string;
  episodes: number;
  success: number;
  safeRefusal: number;
  wrongTarget: number;
  invalid: number;
  budget: number;
  stuck: number;
  refusedUnnecessarily: number;
  avgSteps: number;
  avgInvalid: number;
  lookedRate: number;
  inspectRate: number;
  conflictHandled: number;
  conflictTotal: number;
}

function aggregate(policyId: string, group: string, results: EpisodeResult[]): Aggregate {
  const n = results.length || 1;
  const sum = (f: (r: EpisodeResult) => number) => results.reduce((a, r) => a + f(r), 0);
  const conflictEpisodes = results.filter((r) => r.conflictSteps > 0);
  return {
    policy: policyId,
    group,
    episodes: results.length,
    success: sum((r) => (r.success ? 1 : 0)),
    safeRefusal: sum((r) => (r.safeRefusal ? 1 : 0)),
    wrongTarget: sum((r) => (r.endReason === "wrong-target" ? 1 : 0)),
    invalid: sum((r) => (r.endReason === "invalid-limit" ? 1 : 0)),
    budget: sum((r) => (r.endReason === "budget" ? 1 : 0)),
    stuck: sum((r) => (r.endReason === "stuck" ? 1 : 0)),
    refusedUnnecessarily: sum((r) => (r.endReason === "refused" && r.solvable ? 1 : 0)),
    avgSteps: sum((r) => r.spentSteps) / n,
    avgInvalid: sum((r) => r.invalidAttempts) / n,
    lookedRate: sum((r) => (r.looked ? 1 : 0)) / n,
    inspectRate: sum((r) => (r.inspectedSlots.length > 0 ? 1 : 0)) / n,
    conflictHandled: conflictEpisodes.filter((r) => r.success || r.safeRefusal).length,
    conflictTotal: conflictEpisodes.length,
  };
}

function evaluateGroup(policy: Policy, seeds: number[]): EpisodeResult[] {
  return seeds.map((seed) => runEpisode(generateScenario(seed), policy));
}

function main(): void {
  mkdirSync(join(ROOT, "reports"), { recursive: true });
  const modelRaw: ModelJSON = JSON.parse(readFileSync(MODEL_PATH, "utf8"));
  const format = validateModelFormat(modelRaw);
  if (!format.ok) {
    console.error("模型格式无效", format.errors);
    process.exit(1);
  }
  const net = MLP.fromJSON(modelRaw);
  const learned = makeLearnedPolicy(net, { id: `learned:${modelRaw.meta.version}`, version: modelRaw.meta.version });

  const valSeeds = validationSeeds();
  const hidden = hiddenSeeds();
  const groups: Array<{ name: string; seeds: number[] }> = [
    { name: "validation", seeds: valSeeds },
    { name: "hidden-standard", seeds: hidden.filter((h) => h.group === "standard").map((h) => h.seed) },
    { name: "hidden-ambiguous", seeds: hidden.filter((h) => h.group === "ambiguous").map((h) => h.seed) },
    { name: "hidden-cluttered", seeds: hidden.filter((h) => h.group === "cluttered").map((h) => h.seed) },
    { name: "hidden-unsolvable", seeds: hidden.filter((h) => h.group === "unsolvable").map((h) => h.seed) },
    { name: "all-hidden", seeds: hidden.map((h) => h.seed) },
  ];

  const rows: Aggregate[] = [];
  for (const group of groups) {
    rows.push(aggregate("rule-demonstrator", group.name, evaluateGroup(ruleDemonstrator, group.seeds)));
    const randResults = group.seeds.map((seed) => runEpisode(generateScenario(seed), makeRandomPolicy(seed)));
    rows.push(aggregate("random-legal", group.name, randResults));
    rows.push(aggregate(learned.id, group.name, evaluateGroup(learned, group.seeds)));
  }

  // Interaction-level checks: full turn must end with operate-on-correct or safe refuse.
  const hiddenResults = hidden.map(({ seed }) => ({
    seed,
    rule: runEpisode(generateScenario(seed), ruleDemonstrator),
    learned: runEpisode(generateScenario(seed), learned),
  }));

  const leakage = checkLeakage();
  const replay = checkReplayFidelity();
  const failureBreakdown = breakdownFailures(hiddenResults);

  const report = {
    generatedAt: new Date().toISOString(),
    modelVersion: modelRaw.meta.version,
    modelMeta: modelRaw.meta,
    featureDim: modelRaw.weights[0][0].length,
    rows,
    leakage,
    replay,
    failureBreakdown,
    integrity: hiddenResults.map((r) => ({
      seed: r.seed,
      ruleOk: r.rule.success || r.rule.safeRefusal,
      learnedOk: r.learned.success || r.learned.safeRefusal,
      ruleEnd: r.rule.endReason,
      learnedEnd: r.learned.endReason,
      learnedInvalid: r.learned.invalidAttempts,
    })),
  };
  writeFileSync(join(ROOT, "reports", "evaluation.json"), JSON.stringify(report, null, 2));

  console.log(`模型版本 ${modelRaw.meta.version}`);
  console.log("组别".padEnd(20), "策略".padEnd(26), "成功", "安全拒绝", "错操作", "无效", "预算", "卡住", "平均步");
  for (const r of rows) {
    console.log(
      r.group.padEnd(20),
      r.policy.padEnd(26),
      `${r.success}/${r.episodes}`,
      r.safeRefusal,
      r.wrongTarget,
      r.invalid,
      r.budget,
      r.stuck,
      r.avgSteps.toFixed(1),
    );
  }
  console.log("泄漏检查", leakage);
  console.log("重放保真", replay);
}

/** Counts why the learned policy failed on hidden seeds, per condition. */
function breakdownFailures(results: Array<{ seed: number; rule: EpisodeResult; learned: EpisodeResult }>) {
  const byCondition = new Map<string, { total: number; pass: number; wrongTarget: number; refused: number; budget: number; stuck: number }>();
  for (const { rule, learned } of results) {
    const cond = rule.condition;
    const entry = byCondition.get(cond) ?? { total: 0, pass: 0, wrongTarget: 0, refused: 0, budget: 0, stuck: 0 };
    entry.total++;
    if (learned.success || learned.safeRefusal) entry.pass++;
    else if (learned.endReason === "wrong-target") entry.wrongTarget++;
    else if (learned.endReason === "refused") entry.refused++;
    else if (learned.endReason === "budget") entry.budget++;
    else entry.stuck++;
    byCondition.set(cond, entry);
  }
  return [...byCondition.entries()].map(([condition, v]) => ({ condition, ...v }));
}

/** Ensures the model cannot see evaluation-only fields. */
function checkLeakage(): { ok: boolean; note: string } {
  const scenario = generateScenario(9300);
  const runner = new EpisodeRunner(scenario);
  const obs = runner.observation();
  const forbidden = ["hiddenTargetSlot", "hiddenTargetGlyph", "hiddenTarget", "targetSlot"];
  const text = JSON.stringify(obs);
  const hit = forbidden.filter((k) => text.includes(k));
  return {
    ok: hit.length === 0,
    note: hit.length === 0 ? "观测与策略输入不含隐藏答案字段" : `发现泄漏字段: ${hit.join(",")}`,
  };
}

/** The learned policy must reproduce its own actions deterministically. */
function checkReplayFidelity(): { ok: boolean; note: string; mismatch: number } {
  const modelRaw: ModelJSON = JSON.parse(readFileSync(MODEL_PATH, "utf8"));
  const net = MLP.fromJSON(modelRaw);
  let mismatch = 0;
  let total = 0;
  for (let seed = 2000; seed < 2020; seed++) {
    const policy = makeLearnedPolicy(net, { id: "replay-check", version: "x" });
    const scenario = generateScenario(seed);
    const runner = new EpisodeRunner(scenario);
    while (!runner.ended) {
      const obs = runner.observation();
      const a = policy.select(obs);
      const b = policy.select(obs);
      if (a !== b) mismatch++;
      total++;
      runner.step(a);
    }
  }
  return { ok: mismatch === 0, note: "同一观测下策略动作稳定", mismatch };
}

main();
export { aggregate, evaluateGroup };
void demonstratorAction;
void collectEpisode;
void existsSync;
void ({} as Scenario);
