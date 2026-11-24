import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { generateScenario } from "../src/core/scenario.js";
import { runEpisode } from "../src/core/engine.js";
import { makeArbiter } from "../src/core/arbiter.js";
import { initLinearModel, initModel, macCount, paramCount, MLP } from "../src/core/model.js";
import { validationSeeds, type DemoDataset } from "./dataset.js";
import { datasetToSamples, trainBcDagger } from "./training.js";
import type { EpisodeResult } from "../src/core/types.js";

const ROOT = resolve(import.meta.dirname, "..");

interface Report {
  schema: string;
  thresholds: Record<string, Record<string, number>>;
  selectedTrigger: { threshold: number; mode: "defer" | "gather" | "both"; maxDeferrals: number | null };
  rows: Array<Record<string, number | string>>;
  models: Record<string, { arch: number[]; params: number; macsPerPass: number; artifact: string }>;
  splits: Record<string, number | number[] | string>;
  config: {
    epochs: number;
    lr: number;
    batch: number;
    seed: number;
    daggerRounds: number;
    daggerSeeds: number;
    thresholdCandidates: number[];
  };
  fewShot: FewShotRow[];
}

interface FewShotRow {
  limit: number;
  samples: number;
  model: "linear" | "mlp";
  threshold: number;
  novelPass: number;
  novelWrongOperates: number;
  novelRequests: number;
  hiddenPass: number;
  hiddenWrongOperates: number;
  hiddenRequests: number;
}

function loadModel(file: string): MLP {
  const raw = JSON.parse(readFileSync(join(ROOT, "public", "models", file), "utf8"));
  return MLP.fromJSON(raw);
}

function seedGroups(): Array<{ name: string; seeds: number[] }> {
  const standard = Array.from({ length: 100 }, (_, i) => 9000 + i);
  const ambiguous = Array.from({ length: 100 }, (_, i) => 9300 + i);
  const unsolvable = Array.from({ length: 25 }, (_, i) => 9500 + i);
  const novel = Array.from({ length: 100 }, (_, i) => 3000 + i);
  return [
    { name: "validation", seeds: validationSeeds() },
    { name: "hidden-standard", seeds: standard },
    { name: "hidden-ambiguous", seeds: ambiguous },
    { name: "hidden-unsolvable", seeds: unsolvable },
    { name: "novel-layout", seeds: novel },
  ];
}

function main(): void {
  const report: Report = JSON.parse(readFileSync(join(ROOT, "reports", "demonstrator-policy-experiment.json"), "utf8"));
  const errors: string[] = [];
  if (report.schema !== "demonstrator-policy-experiment/v1") errors.push(`schema=${report.schema}`);

  const linear = loadModel("experiment-linear.json");
  const mlp = loadModel("experiment-mlp.json");
  if (paramCount(linear) !== report.models.linear.params) errors.push("linear 参数数与报告不符");
  if (paramCount(mlp) !== report.models.mlp.params) errors.push("mlp 参数数与报告不符");
  if (macCount(linear) !== report.models.linear.macsPerPass) errors.push("linear MAC 与报告不符");
  if (macCount(mlp) !== report.models.mlp.macsPerPass) errors.push("mlp MAC 与报告不符");
  if (linear.sizes.join("→") !== report.models.linear.arch.join("→")) errors.push("linear 结构不符");
  if (mlp.sizes.join("→") !== report.models.mlp.arch.join("→")) errors.push("mlp 结构不符");

  const nets = { linear, mlp } as const;
  const recomputed = new Map<string, Record<string, number>>();
  for (const group of seedGroups()) {
    for (const kind of ["linear", "mlp"] as const) {
      for (const mode of ["defer", "gather", "both"] as const) {
        const threshold = report.thresholds[kind][mode];
        const config = { threshold, mode, maxDeferrals: null };
        const results: EpisodeResult[] = group.seeds.map((seed) =>
          runEpisode(
            generateScenario(seed),
            makeArbiter(nets[kind], { modelKind: kind, modelVersion: report.models[kind].artifact, config }),
          ),
        );
        const pass = results.filter((r) => r.success || r.safeRefusal).length;
        recomputed.set(`${kind}:${mode}:${group.name}`, {
          pass,
          episodes: results.length,
          requests: results.reduce((a, r) => a + r.demonstrationRequests, 0),
          wrongOperates: results.filter((r) => r.endReason === "wrong-target").length,
        });
      }
    }
  }

  let checked = 0;
  for (const row of report.rows) {
    const model = String(row.model);
    const mode = String(row.mode);
    const group = String(row.group);
    const key = `${model}:${mode}:${group}`;
    const actual = recomputed.get(key);
    if (!actual) {
      errors.push(`报告行 ${key} 无法复算`);
      continue;
    }
    checked++;
    for (const field of ["pass", "episodes", "requests", "wrongOperates"] as const) {
      if (Number(row[field]) !== actual[field]) {
        errors.push(`${key} ${field}: 报告 ${row[field]} vs 复算 ${actual[field]}`);
      }
    }
  }

  // Coverage: every model×mode×group combination present exactly once.
  for (const group of seedGroups()) {
    for (const kind of ["linear", "mlp"] as const) {
      for (const mode of ["defer", "gather", "both"] as const) {
        const key = `${kind}:${mode}:${group.name}`;
        if (!recomputed.has(key)) errors.push(`缺少组合 ${key}`);
      }
    }
  }

  const trigger = report.selectedTrigger;
  const configPath = join(ROOT, "public", "models", "arbiter-config.json");
  const persisted = JSON.parse(readFileSync(configPath, "utf8"));
  if (persisted.trigger.threshold !== trigger.threshold || persisted.trigger.mode !== trigger.mode) {
    errors.push("持久化触发条件与报告不一致");
  }

  // Few-shot sweep is retrained from the same demo dataset and re-run end to end.
  const sweepPath = join(ROOT, "reports", "fewshot-sweep.json");
  const sweep = JSON.parse(readFileSync(sweepPath, "utf8"));
  if (sweep.rows.length !== report.fewShot.length) errors.push("fewshot-sweep 行数与报告不符");
  const dataset: DemoDataset = JSON.parse(readFileSync(join(ROOT, "data", "train-demonstrations.json"), "utf8"));
  const fits = report.config;
  const novel = Array.from({ length: 100 }, (_, i) => 3000 + i);
  const hiddenAll = [
    ...Array.from({ length: 100 }, (_, i) => 9000 + i),
    ...Array.from({ length: 100 }, (_, i) => 9300 + i),
    ...Array.from({ length: 25 }, (_, i) => 9500 + i),
  ];
  let fewChecked = 0;
  for (const expected of report.fewShot) {
    const subset = datasetToSamples(dataset, expected.limit);
    const net = expected.model === "linear" ? initLinearModel(fits.seed) : initModel(fits.seed);
    trainBcDagger(net, subset, {
      epochs: fits.epochs,
      batchSize: fits.batch,
      lr: fits.lr,
      seed: fits.seed,
      l2: 1e-4,
      daggerRounds: fits.daggerRounds,
      daggerSeeds: fits.daggerSeeds,
    });
    const config = { threshold: expected.threshold, mode: "both" as const, maxDeferrals: null };
    const make = (seed: number) => makeArbiter(net, { modelKind: expected.model, modelVersion: "few-shot", config });
    const novelResults = novel.map((seed) => runEpisode(generateScenario(seed), make(seed)));
    const hiddenResults = hiddenAll.map((seed) => runEpisode(generateScenario(seed), make(seed)));
    const pass = (rs: EpisodeResult[]) => rs.filter((r) => r.success || r.safeRefusal).length;
    const actual: FewShotRow = {
      limit: expected.limit,
      samples: subset.length,
      model: expected.model,
      threshold: expected.threshold,
      novelPass: pass(novelResults),
      novelWrongOperates: novelResults.filter((r) => r.endReason === "wrong-target").length,
      novelRequests: novelResults.reduce((a, r) => a + r.demonstrationRequests, 0),
      hiddenPass: pass(hiddenResults),
      hiddenWrongOperates: hiddenResults.filter((r) => r.endReason === "wrong-target").length,
      hiddenRequests: hiddenResults.reduce((a, r) => a + r.demonstrationRequests, 0),
    };
    fewChecked++;
    for (const field of [
      "samples",
      "novelPass",
      "novelWrongOperates",
      "novelRequests",
      "hiddenPass",
      "hiddenWrongOperates",
      "hiddenRequests",
    ] as const) {
      if (expected[field] !== actual[field]) {
        errors.push(`few-shot ${expected.limit}:${expected.model} ${field}: 报告 ${expected[field]} vs 复算 ${actual[field]}`);
      }
    }
    const saved = sweep.rows.find((r: FewShotRow) => r.limit === expected.limit && r.model === expected.model);
    if (!saved) errors.push(`fewshot-sweep 缺少 ${expected.limit}:${expected.model}`);
    else if (saved.novelPass !== actual.novelPass || saved.hiddenPass !== actual.hiddenPass) {
      errors.push(`fewshot-sweep ${expected.limit}:${expected.model} 与复算不符`);
    }
  }

  if (errors.length > 0) {
    console.error(`复核失败（${errors.length} 项）:`);
    for (const e of errors) console.error("  ✗", e);
    process.exit(1);
  }
  console.log(
    `复核通过：${checked} 行指标由模型产物 + 引擎重算完全一致；${fewChecked} 行少量示范由数据集重训重算一致；触发条件与快照一致。`,
  );
}

main();
