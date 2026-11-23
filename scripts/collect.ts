import { writeFileSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { collectDataset, trainSeeds, validationSeeds } from "./dataset.js";
import { scenarioSignature } from "../src/core/scenario.js";
import { generateScenario } from "../src/core/scenario.js";
import { stableHash } from "../src/core/rng.js";

const ROOT = resolve(import.meta.dirname, "..");
const DATA_DIR = join(ROOT, "data");

mkdirSync(DATA_DIR, { recursive: true });

const train = trainSeeds();
const validation = validationSeeds();
const trainSet = new Set(train);
for (const s of validation) {
  if (trainSet.has(s)) throw new Error(`种子泄漏: ${s} 同时出现在训练与验证集`);
}

const trainDataset = collectDataset(train);
writeFileSync(join(DATA_DIR, "train-demonstrations.json"), JSON.stringify(trainDataset));

const valDataset = collectDataset(validation);
writeFileSync(join(DATA_DIR, "validation-demonstrations.json"), JSON.stringify(valDataset));

const sigTrain = stableHash(train.map((s) => scenarioSignature(generateScenario(s))).join("|"));
const sigVal = stableHash(validation.map((s) => scenarioSignature(generateScenario(s))).join("|"));

const manifest = {
  schema: "demonstrations/v1",
  createdAt: new Date().toISOString(),
  train: {
    file: "train-demonstrations.json",
    seeds: train.length,
    episodes: trainDataset.episodes.length,
    steps: trainDataset.episodes.reduce((a, e) => a + e.steps.length, 0),
    success: trainDataset.episodes.filter((e) => e.success).length,
    refusals: trainDataset.episodes.filter((e) => e.endReason === "refused").length,
    hash: trainDataset.hash,
    layoutHash: sigTrain,
  },
  validation: {
    file: "validation-demonstrations.json",
    seeds: validation.length,
    episodes: valDataset.episodes.length,
    steps: valDataset.episodes.reduce((a, e) => a + e.steps.length, 0),
    success: valDataset.episodes.filter((e) => e.success).length,
    refusals: valDataset.episodes.filter((e) => e.endReason === "refused").length,
    hash: valDataset.hash,
    layoutHash: sigVal,
  },
};
writeFileSync(join(DATA_DIR, "manifest.json"), JSON.stringify(manifest, null, 2));

console.log(
  `训练集 ${manifest.train.episodes} 回合 / ${manifest.train.steps} 步，成功 ${manifest.train.success}，拒绝 ${manifest.train.refusals}`,
);
console.log(
  `验证集 ${manifest.validation.episodes} 回合 / ${manifest.validation.steps} 步，成功 ${manifest.validation.success}，拒绝 ${manifest.validation.refusals}`,
);
console.log(`训练/验证布局哈希不同: ${sigTrain !== sigVal}`);
