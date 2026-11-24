import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import {
  ACTION_COUNT,
  FEATURE_DIM,
  NOVEL_LAYOUT_START,
  NOVEL_LAYOUT_END,
  isNovelLayoutSeed,
} from "../src/core/constants.js";
import { generateScenario } from "../src/core/scenario.js";
import { runEpisode } from "../src/core/engine.js";
import { initLinearModel, initModel, macCount, paramCount, type MLP, type TrainSample } from "../src/core/model.js";
import { makeArbiter, type ArbiterConfig, type LowConfidenceMode } from "../src/core/arbiter.js";
import { makeRandomPolicy, makeRuleDemonstrator } from "../src/core/policies.js";
import { stableHash } from "../src/core/rng.js";
import { validationSeeds, type DemoDataset } from "./dataset.js";
import { datasetToSamples, trainBcDagger, type BcDaggerResult } from "./training.js";
import type { EpisodeResult, Policy } from "../src/core/types.js";

const ROOT = resolve(import.meta.dirname, "..");
const DATA_DIR = join(ROOT, "data");
const MODEL_DIR = join(ROOT, "public", "models");
const REPORT_DIR = join(ROOT, "reports");

const MODES: LowConfidenceMode[] = ["defer", "gather", "both"];
const THRESHOLD_CANDIDATES = [0.4, 0.55, 0.7, 0.8, 0.9];

interface CliArgs {
  epochs: number;
  lr: number;
  batch: number;
  seed: number;
  daggerRounds: number;
  daggerSeeds: number;
  fewShotSeeds: string;
  out: string;
}

function parseArgs(): CliArgs {
  const argv = process.argv.slice(2);
  const get = (name: string, fallback: string): string => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
  };
  return {
    epochs: Number(get("epochs", "30")),
    lr: Number(get("lr", "0.003")),
    batch: Number(get("batch", "64")),
    seed: Number(get("seed", "20261124")),
    daggerRounds: Number(get("dagger-rounds", "8")),
    daggerSeeds: Number(get("dagger-seeds", "80")),
    fewShotSeeds: get("few-shot", "10,20,40,80,160,380"),
    out: get("out", join(REPORT_DIR, "demonstrator-policy-experiment.json")),
  };
}

type ModelKind = "linear" | "mlp";

interface Row {
  model: string;
  mode: string;
  group: string;
  threshold: number;
  episodes: number;
  pass: number;
  success: number;
  safeRefusal: number;
  wrongOperates: number;
  unnecessaryRefusals: number;
  budget: number;
  stuck: number;
  requests: number;
  requestRate: number;
  requestTurns: number;
  avgSteps: number;
  avgForwardPasses: number;
  avgFeatureChecks: number;
  avgMultiplyAdds: number;
  avgDecisionMicros: number;
}

function summarize(
  base: { model: string; mode: string; group: string; threshold: number },
  results: EpisodeResult[],
): Row {
  const n = results.length || 1;
  const sum = (f: (r: EpisodeResult) => number) => results.reduce((a, r) => a + f(r), 0);
  const success = sum((r) => (r.success ? 1 : 0));
  const safeRefusal = sum((r) => (r.safeRefusal ? 1 : 0));
  const requests = sum((r) => r.demonstrationRequests);
  return {
    ...base,
    episodes: results.length,
    pass: success + safeRefusal,
    success,
    safeRefusal,
    wrongOperates: sum((r) => (r.endReason === "wrong-target" ? 1 : 0)),
    unnecessaryRefusals: sum((r) => (r.endReason === "refused" && r.solvable ? 1 : 0)),
    budget: sum((r) => (r.endReason === "budget" ? 1 : 0)),
    stuck: sum((r) => (r.endReason === "stuck" ? 1 : 0)),
    requests,
    requestRate: requests / n,
    requestTurns: results.filter((r) => r.demonstrationRequests > 0).length,
    avgSteps: sum((r) => r.spentSteps) / n,
    avgForwardPasses: sum((r) => r.inference.forwardPasses) / n,
    avgFeatureChecks: sum((r) => r.inference.featureChecks) / n,
    avgMultiplyAdds: sum((r) => r.inference.multiplyAdds) / n,
    avgDecisionMicros: sum((r) => r.inference.decisionMicros) / n,
  };
}

/**
 * Per-model threshold selection on the fixed validation split. Primary key is
 * fewest wrong operates; ties break toward fewer teacher requests, then higher
 * pass count. Hidden and novel seeds never participate.
 */
function calibrate(
  net: MLP,
  modelKind: ModelKind,
  mode: LowConfidenceMode,
  seeds: number[],
): { threshold: number; table: Array<{ threshold: number; wrongOperates: number; requests: number; pass: number }> } {
  const table = THRESHOLD_CANDIDATES.map((threshold) => {
    const results = seeds.map((seed) =>
      runEpisode(
        generateScenario(seed),
        makeArbiter(net, { modelKind, modelVersion: "calibrate", config: { threshold, mode, maxDeferrals: null } }),
      ),
    );
    return {
      threshold,
      wrongOperates: results.filter((r) => r.endReason === "wrong-target").length,
      requests: results.reduce((a, r) => a + r.demonstrationRequests, 0),
      pass: results.filter((r) => r.success || r.safeRefusal).length,
    };
  });
  let best = table[0];
  for (const row of table) {
    if (row.wrongOperates < best.wrongOperates) best = row;
    else if (row.wrongOperates === best.wrongOperates && row.requests < best.requests) best = row;
    else if (row.wrongOperates === best.wrongOperates && row.requests === best.requests && row.pass > best.pass) best = row;
  }
  return { threshold: best.threshold, table };
}

function hiddenSeeds(): { standard: number[]; ambiguous: number[]; unsolvable: number[] } {
  return {
    standard: Array.from({ length: 100 }, (_, i) => 9000 + i),
    ambiguous: Array.from({ length: 100 }, (_, i) => 9300 + i),
    unsolvable: Array.from({ length: 25 }, (_, i) => 9500 + i),
  };
}

function novelSeeds(): number[] {
  const out: number[] = [];
  for (let s = NOVEL_LAYOUT_START; s <= NOVEL_LAYOUT_END; s++) if (isNovelLayoutSeed(s)) out.push(s);
  return out;
}

function evaluate(
  factory: (seed: number) => Policy,
  seeds: number[],
  base: { model: string; mode: string; group: string; threshold: number },
): Row {
  const results = seeds.map((seed) => runEpisode(generateScenario(seed), factory(seed)));
  return summarize(base, results);
}

function main(): void {
  const args = parseArgs();
  mkdirSync(REPORT_DIR, { recursive: true });
  mkdirSync(MODEL_DIR, { recursive: true });

  const trainPath = join(DATA_DIR, "train-demonstrations.json");
  if (!existsSync(trainPath)) {
    console.error("缺少示范数据；先运行 npm run demo:collect");
    process.exit(1);
  }
  const dataset: DemoDataset = JSON.parse(readFileSync(trainPath, "utf8"));
  const trainSamples = datasetToSamples(dataset);

  const valSeeds = validationSeeds();
  const hidden = hiddenSeeds();
  const novel = novelSeeds();

  const groups: Array<{ name: string; seeds: number[] }> = [
    { name: "validation", seeds: valSeeds },
    { name: "hidden-standard", seeds: hidden.standard },
    { name: "hidden-ambiguous", seeds: hidden.ambiguous },
    { name: "hidden-unsolvable", seeds: hidden.unsolvable },
    { name: "novel-layout", seeds: novel },
  ];

  const daggerOpts = {
    epochs: args.epochs,
    batchSize: args.batch,
    lr: args.lr,
    seed: args.seed,
    l2: 1e-4,
    daggerRounds: args.daggerRounds,
    daggerSeeds: args.daggerSeeds,
  };
  const linear = initLinearModel(args.seed);
  const mlp = initModel(args.seed);
  const linearFit = trainBcDagger(linear, trainSamples, daggerOpts);
  const mlpFit = trainBcDagger(mlp, trainSamples, daggerOpts);
  const nets: Record<ModelKind, MLP> = { linear, mlp };

  const thresholds: Record<ModelKind, Record<string, number>> = { linear: {}, mlp: {} };
  const calibration: Record<string, unknown> = {};
  for (const kind of ["linear", "mlp"] as ModelKind[]) {
    for (const mode of MODES) {
      const result = calibrate(nets[kind], kind, mode, valSeeds);
      thresholds[kind][mode] = result.threshold;
      calibration[`${kind}:${mode}`] = result;
    }
  }

  const rows: Row[] = [];
  for (const kind of ["linear", "mlp"] as ModelKind[]) {
    for (const mode of MODES) {
      const config: ArbiterConfig = { threshold: thresholds[kind][mode], mode, maxDeferrals: null };
      const factory = (seed: number) =>
        makeArbiter(nets[kind], {
          modelKind: kind,
          modelVersion: modelVersionFor(kind, args),
          config,
        });
      for (const group of groups) {
        rows.push(
          evaluate(factory, group.seeds, {
            model: kind,
            mode,
            group: group.name,
            threshold: config.threshold,
          }),
        );
      }
    }
  }

  // Baselines on the same groups.
  const baselineRows: Row[] = [];
  for (const group of groups) {
    baselineRows.push(
      evaluate(
        () => makeRuleDemonstrator(),
        group.seeds,
        { model: "rule-demonstrator", mode: "none", group: group.name, threshold: 0 },
      ),
    );
    baselineRows.push(
      evaluate(
        (seed) => makeRandomPolicy(seed),
        group.seeds,
        { model: "random-legal", mode: "none", group: group.name, threshold: 0 },
      ),
    );
  }

  // Few-demonstration sweep: how each representation degrades with fewer demos.
  const fewShotLimits = args.fewShotSeeds
    .split(",")
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v) && v > 0);
  const fewShot: Array<{
    limit: number;
    samples: number;
    model: ModelKind;
    threshold: number;
    novelPass: number;
    novelWrongOperates: number;
    novelRequests: number;
    hiddenPass: number;
    hiddenWrongOperates: number;
    hiddenRequests: number;
    fitSeconds: number;
    trainAccuracy: number;
  }> = [];
  for (const limit of fewShotLimits) {
    const subset = datasetToSamples(dataset, limit);
    for (const kind of ["linear", "mlp"] as ModelKind[]) {
      const net = kind === "linear" ? initLinearModel(args.seed) : initModel(args.seed);
      const result = trainBcDagger(net, subset, daggerOpts);
      const cal = calibrate(net, kind, "both", valSeeds);
      const config: ArbiterConfig = { threshold: cal.threshold, mode: "both", maxDeferrals: null };
      const make = (seed: number) => makeArbiter(net, { modelKind: kind, modelVersion: "few-shot", config });
      const novelResults = novel.map((seed) => runEpisode(generateScenario(seed), make(seed)));
      const hiddenAll = [...hidden.standard, ...hidden.ambiguous, ...hidden.unsolvable];
      const hiddenResults = hiddenAll.map((seed) => runEpisode(generateScenario(seed), make(seed)));
      const pass = (rs: EpisodeResult[]) => rs.filter((r) => r.success || r.safeRefusal).length;
      fewShot.push({
        limit,
        samples: subset.length,
        model: kind,
        threshold: cal.threshold,
        novelPass: pass(novelResults),
        novelWrongOperates: novelResults.filter((r) => r.endReason === "wrong-target").length,
        novelRequests: novelResults.reduce((a, r) => a + r.demonstrationRequests, 0),
        hiddenPass: pass(hiddenResults),
        hiddenWrongOperates: hiddenResults.filter((r) => r.endReason === "wrong-target").length,
        hiddenRequests: hiddenResults.reduce((a, r) => a + r.demonstrationRequests, 0),
        fitSeconds: result.seconds,
        trainAccuracy: result.accuracy,
      });
    }
  }

  const costBenchmark = benchmarkForward(linear, mlp);

  const selectedConfig: ArbiterConfig = {
    threshold: thresholds.linear.both,
    mode: "both",
    maxDeferrals: null,
  };
  const datasetHash = dataset.hash;
  const splitHash = stableHash({
    train: dataset.seeds.length,
    valSeeds,
    novel,
    hiddenStandard: hidden.standard.length,
    hiddenAmbiguous: hidden.ambiguous.length,
    hiddenUnsolvable: hidden.unsolvable.length,
  });

  const payload = {
    schema: "demonstrator-policy-experiment/v1",
    generatedAt: new Date().toISOString(),
    evidence: {
      runner: "scripts/experiment.ts",
      modelRebuild: "npm run demo:experiment",
      verification: "npm run demo:verify:experiment",
      note: "所有动作成功率与推理成本均由 runEpisode 完整回合重算；计时列仅作本机参考，不参与断言。",
    },
    config: {
      epochs: args.epochs,
      lr: args.lr,
      batch: args.batch,
      seed: args.seed,
      daggerRounds: args.daggerRounds,
      daggerSeeds: args.daggerSeeds,
      featureDim: FEATURE_DIM,
      actionCount: ACTION_COUNT,
      trainSamples: trainSamples.length,
      trainEpisodes: dataset.episodes.length,
      thresholdCandidates: THRESHOLD_CANDIDATES,
    },
    splits: {
      train: dataset.seeds.length,
      validation: valSeeds.length,
      hiddenStandard: hidden.standard.length,
      hiddenAmbiguous: hidden.ambiguous.length,
      hiddenUnsolvable: hidden.unsolvable.length,
      novelLayout: novel.length,
      novelRange: [NOVEL_LAYOUT_START, NOVEL_LAYOUT_END],
      datasetHash,
      splitHash,
    },
    models: {
      linear: {
        arch: linear.sizes,
        params: paramCount(linear),
        macsPerPass: macCount(linear),
        fit: linearFit,
        artifact: "models/experiment-linear.json",
      },
      mlp: {
        arch: mlp.sizes,
        params: paramCount(mlp),
        macsPerPass: macCount(mlp),
        fit: mlpFit,
        artifact: "models/experiment-mlp.json",
      },
    },
    thresholds,
    calibration,
    selectedTrigger: selectedConfig,
    rows,
    baselines: baselineRows,
    fewShot,
    costBenchmark,
  };

  writeFileSync(args.out, JSON.stringify(payload, null, 2));
  writeFileSync(
    join(REPORT_DIR, "fewshot-sweep.json"),
    JSON.stringify(
      {
        schema: "fewshot-sweep/v1",
        generatedAt: payload.generatedAt,
        evidence: payload.evidence,
        config: payload.config,
        splits: payload.splits,
        rows: fewShot,
      },
      null,
      2,
    ),
  );
  writeFileSync(join(MODEL_DIR, "experiment-linear.json"), JSON.stringify(netJSON(linear, "local-linear-experiment", args, linearFit), null, 2));
  writeFileSync(join(MODEL_DIR, "experiment-mlp.json"), JSON.stringify(netJSON(mlp, "local-mlp-experiment", args, mlpFit), null, 2));
  writeFileSync(
    join(MODEL_DIR, "arbiter-config.json"),
    JSON.stringify(
      {
        schema: "arbiter-config/v1",
        generatedAt: payload.generatedAt,
        trigger: selectedConfig,
        thresholds,
        selectedModels: {
          linear: "experiment-linear.json",
          mlp: "experiment-mlp.json",
        },
        experiment: "reports/demonstrator-policy-experiment.json",
        note: "低置信触发条件；由 demo:experiment 标定并在图形界面加载。",
      },
      null,
      2,
    ),
  );

  console.log(`实验写入 ${args.out}`);
  console.log("模型  模式   组别              通过    错操作  请求  前向/回合  微秒/回合");
  for (const row of [...rows, ...baselineRows]) {
    console.log(
      row.model.padEnd(16),
      row.mode.padEnd(7),
      row.group.padEnd(18),
      `${row.pass}/${row.episodes}`.padEnd(8),
      String(row.wrongOperates).padEnd(7),
      String(row.requests).padEnd(5),
      row.avgForwardPasses.toFixed(1).padEnd(10),
      row.avgDecisionMicros.toFixed(1),
    );
  }
  console.log("少量示范扫描：");
  for (const row of fewShot) {
    console.log(
      `  回合≤${String(row.limit).padEnd(4)} ${row.model.padEnd(7)} 新布局 通过 ${row.novelPass}/100 错操作 ${row.novelWrongOperates} 请求 ${row.novelRequests} | 隐藏 通过 ${row.hiddenPass}/225 错操作 ${row.hiddenWrongOperates} 请求 ${row.hiddenRequests}`,
    );
  }
}

function modelVersionFor(kind: ModelKind, args: CliArgs): string {
  return `${kind}-exp-${args.seed}`;
}

function netJSON(net: MLP, id: string, args: CliArgs, fitResult: BcDaggerResult): {
  meta: Record<string, unknown>;
  weights: number[][][];
  biases: number[][];
} {
  return {
    meta: {
      id,
      version: `${id}-${stableHash({ w: net.weights, seed: args.seed }).slice(0, 8)}`,
      featureVersion: "obs-v2",
      actionVersion: "act-v2",
      arch: net.sizes,
      seed: args.seed,
      epochs: args.epochs,
      daggerRounds: args.daggerRounds,
      lr: args.lr,
      batch: args.batch,
      baseSamples: fitResult.baseSamples,
      finalSamples: fitResult.finalSamples,
      loss: fitResult.loss,
      accuracy: fitResult.accuracy,
    },
    weights: net.weights,
    biases: net.biases,
  };
}

function benchmarkForward(linear: MLP, mlp: MLP): Array<{ model: string; macs: number; microsPerPass: number; passes: number }> {
  const input = new Array(FEATURE_DIM).fill(0.5);
  const measure = (net: MLP) => {
    for (let i = 0; i < 200; i++) net.forward(input);
    const t0 = performance.now();
    const n = 5000;
    for (let i = 0; i < n; i++) net.forward(input);
    return ((performance.now() - t0) * 1000) / n;
  };
  return [
    { model: "linear", macs: macCount(linear), microsPerPass: measure(linear), passes: 5000 },
    { model: "mlp", macs: macCount(mlp), microsPerPass: measure(mlp), passes: 5000 },
  ];
}

main();
