import { EpisodeRunner } from "../src/core/engine.js";
import { demonstratorAction, makeLearnedPolicy } from "../src/core/policies.js";
import { generateScenario } from "../src/core/scenario.js";
import { trainModel, type MLP, type TrainSample, type TrainConfig } from "../src/core/model.js";
import type { DemoDataset } from "./dataset.js";
import { daggerSeeds as daggerSeedList } from "./dataset.js";

export function datasetToSamples(dataset: DemoDataset, limit: number | null = null): TrainSample[] {
  const episodes = limit === null ? dataset.episodes : dataset.episodes.slice(0, limit);
  const samples: TrainSample[] = [];
  for (const ep of episodes) {
    for (const step of ep.steps) {
      if (!step.legalMask[step.action]) continue;
      samples.push({ x: step.x, y: step.action, weight: 1 });
    }
  }
  return samples;
}

/**
 * DAgger round shared by every softmax model: roll out the *student* on fresh
 * training seeds, label every visited state with the demonstrator, and mix in
 * 20% teacher actions so the rollout stays corrective.
 */
export function daggerSamples(net: MLP, round: number, seedBase: number, count: number): TrainSample[] {
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
      const action = nextRand() < 0.2 ? teacher : studentAction;
      if (!runner.step(obs.legalActions[action]?.legal ? action : teacher)) break;
    }
  }
  return samples;
}

export interface BcDaggerOptions {
  epochs: number;
  batchSize: number;
  lr: number;
  seed: number;
  l2: number;
  daggerRounds: number;
  daggerSeeds: number;
  onRound?: (label: string, samples: number, loss: number, accuracy: number) => void;
}
export interface BcDaggerResult {
  loss: number;
  accuracy: number;
  baseSamples: number;
  finalSamples: number;
  daggerAdded: number;
  seconds: number;
}

/** Behavior cloning followed by DAgger, identical for linear and hidden-layer MLPs. */
export function trainBcDagger(net: MLP, base: TrainSample[], opts: BcDaggerOptions): BcDaggerResult {
  const start = performance.now();
  const first = trainModel(net, base, config(opts, opts.epochs, opts.seed));
  opts.onRound?.("bc", base.length, first.losses.at(-1) ?? 0, first.accuracy);
  let merged = base;
  let added = 0;
  let last = first;
  for (let round = 0; round < opts.daggerRounds; round++) {
    const extra = daggerSamples(net, round, opts.seed, opts.daggerSeeds);
    added += extra.length;
    merged = [...merged, ...extra];
    last = trainModel(net, merged, config({ ...opts, lr: opts.lr / 2 }, Math.max(4, Math.floor(opts.epochs / 3)), opts.seed + round + 1));
    opts.onRound?.(`dagger-${round + 1}`, merged.length, last.losses.at(-1) ?? 0, last.accuracy);
  }
  return {
    loss: last.losses.at(-1) ?? 0,
    accuracy: last.accuracy,
    baseSamples: base.length,
    finalSamples: merged.length,
    daggerAdded: added,
    seconds: (performance.now() - start) / 1000,
  };
}

function config(opts: BcDaggerOptions, epochs: number, seed: number): TrainConfig {
  return { epochs, batchSize: opts.batchSize, lr: opts.lr, seed, l2: opts.l2 };
}
