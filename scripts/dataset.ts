import { FEATURE_DIM } from "../src/core/constants.js";
import { generateScenario, seedLabel, scenarioSignature } from "../src/core/scenario.js";
import { EpisodeRunner } from "../src/core/engine.js";
import { demonstratorAction } from "../src/core/policies.js";
import { stableHash } from "../src/core/rng.js";

export interface DemoStep {
  seed: number;
  step: number;
  x: number[];
  action: number;
  legalMask: number[];
  endReason: string | null;
  conflict: boolean;
}

export interface DemoEpisode {
  seed: number;
  condition: string;
  solvable: boolean;
  success: boolean;
  endReason: string | null;
  actions: number[];
  steps: DemoStep[];
  signature: string;
}

export interface DemoDataset {
  schema: "demonstrations/v1";
  featureVersion: string;
  actionVersion: string;
  createdAt: string;
  featureDim: number;
  seeds: number[];
  episodes: DemoEpisode[];
  hash: string;
}

export function collectEpisode(seed: number): DemoEpisode {
  const scenario = generateScenario(seed);
  const runner = new EpisodeRunner(scenario);
  const steps: DemoStep[] = [];
  while (!runner.ended) {
    const obs = runner.observation();
    const action = demonstratorAction(obs);
    const record = runner.step(action);
    if (!record) break;
    steps.push({
      seed,
      step: record.index,
      x: obs.vector,
      action,
      legalMask: obs.legalActions.map((l) => (l.legal ? 1 : 0)),
      endReason: runner.endReason,
      conflict: record.conflict,
    });
  }
  return {
    seed,
    condition: scenario.condition,
    solvable: scenario.solvable,
    success: runner.endReason === "success",
    endReason: runner.endReason,
    actions: steps.map((s) => s.action),
    steps,
    signature: stableHash(scenarioSignature(scenario)),
  };
}

export function collectDataset(seeds: number[]): DemoDataset {
  const episodes = seeds.map(collectEpisode);
  const base = {
    schema: "demonstrations/v1" as const,
    featureVersion: "obs-v2",
    actionVersion: "act-v2",
    createdAt: new Date().toISOString(),
    featureDim: FEATURE_DIM,
    seeds,
    episodes,
  };
  return { ...base, hash: stableHash(base) };
}

export function trainSeeds(): number[] {
  const seeds: number[] = [];
  for (let s = 1000; s <= 1279; s++) seeds.push(s);
  for (let s = 1600; s <= 1699; s++) seeds.push(s);
  return seeds;
}

/** Fixed pool of DAgger training seeds covering every solvable layout family. */
export const DAGGER_SEED_POOL: number[] = (() => {
  const out: number[] = [];
  // standard / cluttered / ambiguous shapes live in the ordinary train range
  for (let s = 1000; s <= 1279; s++) out.push(s);
  // explicit refusal layouts
  for (let s = 1600; s <= 1699; s++) out.push(s);
  return out;
})();

/**
 * Seeds used for DAgger rollouts. Always drawn from the training pool — never
 * validation or hidden seeds.
 */
export function daggerSeeds(round: number, count: number, offset: number): number[] {
  const rotated = [...DAGGER_SEED_POOL.slice(offset % 7), ...DAGGER_SEED_POOL.slice(0, offset % 7)];
  const start = (round * count + offset * 13) % rotated.length;
  const out: number[] = [];
  for (let i = 0; i < count; i++) out.push(rotated[(start + i) % rotated.length]);
  return out;
}

export function validationSeeds(): number[] {
  const seeds: number[] = [];
  for (let s = 2000; s <= 2079; s++) seeds.push(s);
  for (let s = 2600; s <= 2619; s++) seeds.push(s);
  return seeds;
}

export function hiddenSeeds(): Array<{ seed: number; group: string }> {
  const out: Array<{ seed: number; group: string }> = [];
  for (let s = 9000; s <= 9099; s++) out.push({ seed: s, group: "standard" });
  for (let s = 9300; s <= 9399; s++) out.push({ seed: s, group: "ambiguous" });
  for (let s = 9400; s <= 9499; s++) out.push({ seed: s, group: "cluttered" });
  for (let s = 9500; s <= 9524; s++) out.push({ seed: s, group: "unsolvable" });
  return out;
}

export { seedLabel };
