import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  FEATURE_VERSION,
  ACTION_VERSION,
  DATASET_VERSION,
  ACTION_COUNT,
  FEATURE_DIM,
} from "../src/core/constants.js";
import { EpisodeRunner } from "../src/core/engine.js";
import { demonstratorAction, makeLearnedPolicy } from "../src/core/policies.js";
import { generateScenario } from "../src/core/scenario.js";
import { initModel, trainModel, MLP, modelVersion, type TrainSample, type ModelJSON } from "../src/core/model.js";
import { stableHash } from "../src/core/rng.js";
import type { DemoDataset } from "./dataset.js";
import { daggerSeeds as daggerSeedList } from "./dataset.js";

const ROOT = resolve(import.meta.dirname, "..");
const DATA_DIR = join(ROOT, "data");
const MODEL_DIR = join(ROOT, "public", "models");

interface Args {
  dataset: string;
  out: string;
  epochs: number;
  lr: number;
  batch: number;
  seed: number;
  daggerRounds: number;
  daggerSeeds: number;
}

function parseArgs(): Args {
  const argv = process.argv.slice(2);
  const get = (name: string, fallback: string): string => {
    const i = argv.indexOf(`--${name}`);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
  };
  return {
    dataset: get("dataset", join(DATA_DIR, "train-demonstrations.json")),
    out: get("out", join(MODEL_DIR, "learned-model.json")),
    epochs: Number(get("epochs", "30")),
    lr: Number(get("lr", "0.003")),
    batch: Number(get("batch", "64")),
    seed: Number(get("seed", "12345")),
    daggerRounds: Number(get("dagger-rounds", "3")),
    daggerSeeds: Number(get("dagger-seeds", "80")),
  };
}

function datasetToSamples(dataset: DemoDataset): TrainSample[] {
  const samples: TrainSample[] = [];
  for (const ep of dataset.episodes) {
    for (const step of ep.steps) {
      if (!step.legalMask[step.action]) continue;
      samples.push({ x: step.x, y: step.action, weight: 1 });
    }
  }
  return samples;
}

/**
 * DAgger: roll out the *student* on fresh training seeds, label every visited
 * state with the demonstrator, and refresh the training set. Student actions are
 * mixed with the teacher's so the rollout explores the states the model would
 * reach, including where it previously mishandled a smudged mark.
 */
function daggerSamples(net: MLP, round: number, seedBase: number, count: number): TrainSample[] {
  const samples: TrainSample[] = [];
  const seeds = daggerSeedList(round, count, seedBase);
  const student = makeLearnedPolicy(net, { id: "dagger-student", version: "tmp" });
  let rand = (Math.imul(round + 1, 2654435761) + seedBase) >>> 0;
  const nextRand = () => {
    rand = (Math.imul(rand, 1664525) + 1013904223) >>> 0;
    return rand / 4294967296;
  };
  for (const seed of seeds) {
    const scenario = generateScenario(seed);
    const runner = new EpisodeRunner(scenario);
    let guard = 0;
    while (!runner.ended && guard++ < 80) {
      const obs = runner.observation();
      const teacher = demonstratorAction(obs);
      if (obs.legalActions[teacher]?.legal) samples.push({ x: obs.vector, y: teacher, weight: 1 });
      const studentAction = student.select(obs);
      // 20% teacher takeover keeps the rollout on-policy but corrective.
      const action = nextRand() < 0.2 ? teacher : studentAction;
      if (!runner.step(obs.legalActions[action]?.legal ? action : teacher)) break;
    }
  }
  return samples;
}

function main(): void {
  const args = parseArgs();
  if (!existsSync(args.dataset)) {
    console.error(`缺少示范数据: ${args.dataset}\n先运行 npm run demo:collect`);
    process.exit(1);
  }
  mkdirSync(MODEL_DIR, { recursive: true });
  const dataset: DemoDataset = JSON.parse(readFileSync(args.dataset, "utf8"));
  const datasetSeeds = new Set(dataset.seeds);
  const allSamples = datasetToSamples(dataset);
  console.log(
    `数据集 ${dataset.hash} 种子=${dataset.episodes.length} 步样本=${allSamples.length} 特征维=${FEATURE_DIM} 动作数=${ACTION_COUNT}`,
  );
  if (dataset.featureVersion !== FEATURE_VERSION || dataset.actionVersion !== ACTION_VERSION) {
    console.error("数据集版本与当前内核不匹配", dataset.featureVersion, dataset.actionVersion);
    process.exit(1);
  }

  const net = initModel(args.seed);
  let history = trainModel(net, allSamples, {
    epochs: args.epochs,
    batchSize: args.batch,
    lr: args.lr,
    seed: args.seed,
    l2: 1e-4,
  }, (epoch, loss, acc) => {
    if (epoch % 5 === 0 || epoch === args.epochs - 1) {
      console.log(`  epoch ${String(epoch).padStart(2)} loss=${loss.toFixed(4)} acc=${(acc * 100).toFixed(1)}%`);
    }
  });
  console.log(`行为克隆完成 loss=${history.losses.at(-1)!.toFixed(4)} 训练准确率=${(history.accuracy * 100).toFixed(1)}%`);

  for (let round = 0; round < args.daggerRounds; round++) {
    const extra = daggerSamples(net, round, args.seed, args.daggerSeeds);
    const merged = [...allSamples, ...extra];
    console.log(`DAgger 第 ${round + 1} 轮：追加 ${extra.length} 条教师标注样本，合计 ${merged.length}`);
    history = trainModel(net, merged, {
      epochs: Math.max(4, Math.floor(args.epochs / 3)),
      batchSize: args.batch,
      lr: args.lr / 2,
      seed: args.seed + round + 1,
      l2: 1e-4,
    });
    console.log(`  准确率=${(history.accuracy * 100).toFixed(1)}%`);
  }

  const payload = net.toJSON({} as never);
  const version = modelVersion(net, {
    id: "local-bc-dagger",
    featureVersion: FEATURE_VERSION,
    actionVersion: ACTION_VERSION,
    datasetVersion: DATASET_VERSION,
    createdAt: new Date().toISOString(),
    seeds: dataset.seeds,
    arch: [FEATURE_DIM, 96, 96, ACTION_COUNT],
    trainSamples: allSamples.length,
    accuracy: history.accuracy,
    loss: history.losses.at(-1) ?? 0,
    notes: `行为克隆 + ${args.daggerRounds} 轮 DAgger；样本 ${allSamples.length}；训练准确率 ${(history.accuracy * 100).toFixed(1)}%`,
  });
  const json: ModelJSON = {
    meta: {
      id: "local-bc-dagger",
      version,
      featureVersion: FEATURE_VERSION,
      actionVersion: ACTION_VERSION,
      datasetVersion: DATASET_VERSION,
      createdAt: new Date().toISOString(),
      seeds: dataset.seeds,
      arch: [FEATURE_DIM, 96, 96, ACTION_COUNT],
      trainSamples: allSamples.length,
      accuracy: history.accuracy,
      loss: history.losses.at(-1) ?? 0,
      notes: `行为克隆 + ${args.daggerRounds} 轮 DAgger；数据集 ${dataset.hash}`,
    },
    weights: payload.weights,
    biases: payload.biases,
  };
  writeFileSync(args.out, JSON.stringify(json));
  const manifestPath = join(MODEL_DIR, "manifest.json");
  const manifest = existsSync(manifestPath) ? JSON.parse(readFileSync(manifestPath, "utf8")) : { models: [] };
  manifest.models = [
    { version, file: "learned-model.json", meta: json.meta, datasetHash: dataset.hash },
    ...(manifest.models ?? []).filter((m: { version: string }) => m.version !== version),
  ].slice(0, 5);
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  writeFileSync(
    join(ROOT, "reports", "training-summary.json"),
    JSON.stringify(
      {
        datasetHash: dataset.hash,
        datasetSeeds: dataset.seeds.length,
        trainSamples: allSamples.length,
        modelVersion: version,
        accuracy: history.accuracy,
        trainSeeds: [...datasetSeeds].slice(0, 5),
      },
      null,
      2,
    ),
  );
  console.log(`模型已写入 ${args.out}\n版本 ${version}`);
}

main();
