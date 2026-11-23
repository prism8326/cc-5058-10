import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { generateScenario } from "../src/core/scenario.js";
import { runEpisode } from "../src/core/engine.js";
import { makeReplayPolicy } from "../src/core/policies.js";
import type { DemoDataset } from "./dataset.js";

const ROOT = resolve(import.meta.dirname, "..");

for (const file of ["train-demonstrations.json", "validation-demonstrations.json"]) {
  const dataset: DemoDataset = JSON.parse(readFileSync(join(ROOT, "data", file), "utf8"));
  const bySeed = new Map(dataset.episodes.map((e) => [e.seed, e.actions]));
  let mismatch = 0;
  let checked = 0;
  for (const ep of dataset.episodes) {
    if (ep.actions.length === 0) continue;
    const scenario = generateScenario(ep.seed);
    const replay = runEpisode(scenario, makeReplayPolicy("replay", bySeed));
    checked++;
    const sameEnd = replay.endReason === ep.endReason;
    const sameActions = replay.steps.map((s) => s.action).join(",") === ep.actions.join(",");
    if (!sameEnd || !sameActions) {
      mismatch++;
      console.log(`  ✗ seed ${ep.seed}: 记录 ${ep.endReason} / 重放 ${replay.endReason}`);
    }
  }
  console.log(`${file}: ${checked} 回合重放，不一致 ${mismatch}`);
  if (mismatch > 0) process.exitCode = 1;
}
