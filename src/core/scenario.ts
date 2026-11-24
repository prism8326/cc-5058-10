import {
  GRID,
  MAX_FIXTURES,
  TRAIN_SEED_START,
  TRAIN_SEED_END,
  TRAIN_REFUSAL_START,
  TRAIN_REFUSAL_END,
  DEV_SEED_START,
  DEV_SEED_END,
  VAL_SEED_START,
  VAL_SEED_END,
  VAL_REFUSAL_START,
  VAL_REFUSAL_END,
  hiddenConditionForSeed,
  isNovelLayoutSeed,
} from "./constants";
import { mulberry32 } from "./rng";
import { cellIndex, nearestApproach, type NavGrid } from "./nav";
import type {
  ColorId,
  Condition,
  Fixture,
  GlyphId,
  GridPos,
  Scenario,
  ShapeId,
} from "./types";

type Rng = ReturnType<typeof mulberry32>;

export interface ScenarioSpec {
  condition: Condition;
  clutter: number;
  spacing: number;
  nearMiss: boolean;
  duplicatePair: boolean;
  noMatch: boolean;
  /** Novel-layout family: different spawn corner and a denser, tighter scene. */
  novelLayout: boolean;
}

export function scenarioSpec(seed: number): ScenarioSpec {
  const hidden = hiddenConditionForSeed(seed);
  const novel = isNovelLayoutSeed(seed);
  const local = seed % 29;
  const refusalTraining =
    (seed >= TRAIN_REFUSAL_START && seed <= TRAIN_REFUSAL_END) ||
    (seed >= VAL_REFUSAL_START && seed <= VAL_REFUSAL_END);
  if (refusalTraining) {
    // A device pair that is observably identical: the only correct answer is to
    // refuse. These scenes teach the model that action even though it never
    // appears in solvable training layouts.
    return {
      condition: "unsolvable",
      clutter: 4,
      spacing: 2,
      nearMiss: false,
      duplicatePair: true,
      noMatch: false,
      novelLayout: false,
    };
  }
  if (hidden === "unsolvable" || (!hidden && seed === 900)) {
    return {
      condition: "unsolvable",
      clutter: 5,
      spacing: 2,
      nearMiss: false,
      duplicatePair: true,
      noMatch: true,
      novelLayout: false,
    };
  }
  if (hidden === "ambiguous" || (!hidden && local === 7)) {
    return {
      condition: "ambiguous",
      clutter: 4,
      spacing: 2,
      nearMiss: true,
      duplicatePair: false,
      noMatch: false,
      novelLayout: false,
    };
  }
  if (hidden === "cluttered" || (!hidden && (local === 11 || local === 19))) {
    return {
      condition: "cluttered",
      clutter: 11,
      spacing: 1,
      nearMiss: local % 2 === 0,
      duplicatePair: false,
      noMatch: false,
      novelLayout: false,
    };
  }
  if (novel) {
    // Same observable schema, different geometry: mirrored spawn and more
    // obstacles. This is the distribution shift the experiment probes.
    return {
      condition: "standard",
      clutter: 15,
      spacing: 1,
      nearMiss: local % 3 === 0,
      duplicatePair: false,
      noMatch: false,
      novelLayout: true,
    };
  }
  return {
    condition: "standard",
    clutter: 4,
    spacing: 2,
    nearMiss: local % 3 === 0,
    duplicatePair: false,
    noMatch: false,
    novelLayout: false,
  };
}

export function seedLabel(seed: number): "train" | "dev" | "validation" | "novel" | "hidden-eval" | "manual" {
  if (seed >= TRAIN_SEED_START && seed <= TRAIN_SEED_END) return "train";
  if (seed >= DEV_SEED_START && seed <= DEV_SEED_END) return "dev";
  if (seed >= VAL_SEED_START && seed <= VAL_SEED_END) return "validation";
  if (isNovelLayoutSeed(seed)) return "novel";
  if (hiddenConditionForSeed(seed)) return "hidden-eval";
  return "manual";
}

// ---------------------------------------------------------------------------

const inBounds = (x: number, y: number): boolean => x >= 1 && y >= 1 && x < GRID - 1 && y < GRID - 1;

function emptyWalls(): boolean[] {
  return new Array(GRID * GRID).fill(false);
}

function borderWalls(walls: boolean[]): void {
  for (let i = 0; i < GRID; i++) {
    walls[cellIndex(i, 0, GRID)] = true;
    walls[cellIndex(i, GRID - 1, GRID)] = true;
    walls[cellIndex(0, i, GRID)] = true;
    walls[cellIndex(GRID - 1, i, GRID)] = true;
  }
}

function carveBlobs(walls: boolean[], rng: Rng, count: number, spawn: GridPos): void {
  let placed = 0;
  let guard = 0;
  while (placed < count && guard < 800) {
    guard++;
    const w = rng.range(1, 3);
    const h = rng.range(1, 2);
    const x0 = rng.range(2, GRID - 2 - w);
    const y0 = rng.range(2, GRID - 2 - h);
    // Keep a 5×4 corridor around the spawn clear. For the legacy spawn (6,10)
    // this is exactly the original rows 8-11 / cols 4-8 rectangle.
    const overlapsSpawn = x0 <= spawn.x + 2 && x0 + w >= spawn.x - 2 && y0 <= spawn.y + 1 && y0 + h >= spawn.y - 2;
    if (overlapsSpawn) continue;
    for (let y = y0; y < y0 + h; y++) for (let x = x0; x < x0 + w; x++) walls[cellIndex(x, y, GRID)] = true;
    placed++;
  }
}

function isFree(walls: boolean[], x: number, y: number, occupied: Set<number>): boolean {
  if (x < 2 || y < 2 || x > GRID - 3 || y > GRID - 3) return false;
  if (walls[cellIndex(x, y, GRID)]) return false;
  return !occupied.has(cellIndex(x, y, GRID));
}

function pickCells(walls: boolean[], rng: Rng, spacing: number, spawn: GridPos): GridPos[] {
  const occupied = new Set<number>([cellIndex(spawn.x, spawn.y, GRID)]);
  const cells: GridPos[] = [];
  let guard = 0;
  while (cells.length < MAX_FIXTURES && guard < 8000) {
    guard++;
    const x = rng.range(2, GRID - 3);
    const y = rng.range(2, GRID - 3);
    if (!isFree(walls, x, y, occupied)) continue;
    if (Math.abs(x - spawn.x) + Math.abs(y - spawn.y) < 3) continue;
    if (cells.some((c) => Math.abs(c.x - x) + Math.abs(c.y - y) < spacing)) continue;
    occupied.add(cellIndex(x, y, GRID));
    cells.push({ x, y });
  }
  // Deterministic fallback: scan for free cells, ignoring spacing.
  for (let y = 2; y <= GRID - 3 && cells.length < MAX_FIXTURES; y++) {
    for (let x = 2; x <= GRID - 3 && cells.length < MAX_FIXTURES; x++) {
      if (cells.some((c) => c.x === x && c.y === y)) continue;
      if (x === spawn.x && y === spawn.y) continue;
      if (occupied.has(cellIndex(x, y, GRID))) continue;
      walls[cellIndex(x, y, GRID)] = false;
      cells.push({ x, y });
    }
  }
  return cells;
}

function carveCorridor(walls: boolean[], from: GridPos, to: GridPos): void {
  let x = from.x;
  let y = from.y;
  let guard = 0;
  while ((x !== to.x || y !== to.y) && guard++ < 64) {
    if (x !== to.x) x += Math.sign(to.x - x);
    else if (y !== to.y) y += Math.sign(to.y - y);
    if (inBounds(x, y)) walls[cellIndex(x, y, GRID)] = false;
  }
}

/** Walls around each fixture cell are removed so it can always be approached. */
function ensureApproachable(walls: boolean[], spawn: GridPos, fixtures: Fixture[]): void {
  const nav: NavGrid = { gridSize: GRID, walls, blocked: new Set() };
  for (let round = 0; round < 4; round++) {
    let missing: Fixture | null = null;
    for (const f of fixtures) {
      if (!nearestApproach(nav, spawn, f.pos)) {
        missing = f;
        break;
      }
    }
    if (!missing) return;
    carveCorridor(walls, spawn, missing.pos);
  }
}

function allTriples(): Array<{ color: ColorId; shape: ShapeId; glyph: GlyphId }> {
  const out: Array<{ color: ColorId; shape: ShapeId; glyph: GlyphId }> = [];
  for (let c = 0; c < 3; c++)
    for (let s = 0; s < 3; s++)
      for (let g = 0; g < 3; g++) out.push({ color: c as ColorId, shape: s as ShapeId, glyph: g as GlyphId });
  return out;
}

const tripleKey = (t: { color: number; shape: number; glyph: number }): string =>
  `${t.color}:${t.shape}:${t.glyph}`;

export function generateScenario(seed: number): Scenario {
  const spec = scenarioSpec(seed);
  const rng = mulberry32((Math.imul(seed, 2654435761) ^ 0x5f3759df) >>> 0);
  const walls = emptyWalls();
  borderWalls(walls);
  // Novel-layout seeds spawn in the opposite corner; everything downstream
  // (nav, observation encoding) is unchanged.
  const spawn: GridPos = spec.novelLayout ? { x: GRID - 7, y: 2 } : { x: 6, y: 10 };
  carveBlobs(walls, rng, spec.clutter, spawn);
  walls[cellIndex(spawn.x, spawn.y, GRID)] = false;
  for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
    const x = spawn.x + dx;
    const y = spawn.y + dy;
    if (inBounds(x, y)) walls[cellIndex(x, y, GRID)] = false;
  }

  const cells = pickCells(walls, rng, spec.spacing, spawn);
  for (const c of cells) walls[cellIndex(c.x, c.y, GRID)] = false;

  const targetSlot = rng.int(MAX_FIXTURES);
  const goal = {
    color: rng.int(3) as ColorId,
    shape: rng.int(3) as ShapeId,
    glyph: rng.int(3) as GlyphId,
  };

  const deviceSlots = new Set<number>();
  if (!spec.noMatch) deviceSlots.add(targetSlot);
  const others = rng.shuffle([0, 1, 2, 3, 4, 5].filter((s) => s !== targetSlot));
  const extraDevices = spec.condition === "ambiguous" ? 3 : 2 + rng.int(2);
  while (deviceSlots.size < Math.min(extraDevices + (spec.noMatch ? 1 : 0), MAX_FIXTURES - 1)) {
    const s = others.pop();
    if (s === undefined) break;
    deviceSlots.add(s);
  }

  const triples = new Map<number, { color: ColorId; shape: ShapeId; glyph: GlyphId }>();
  if (!spec.noMatch) triples.set(targetSlot, { ...goal });

  const used = new Set<string>(spec.noMatch ? [] : [tripleKey(goal)]);
  const pool = allTriples();

  // Near-miss device: same visible colour+shape as the goal, different glyph.
  let nearMissSlot: number | null = null;
  if (spec.nearMiss && deviceSlots.size > 1) {
    const candidates = [...deviceSlots].filter((s) => s !== targetSlot).sort((a, b) => a - b);
    if (candidates.length > 0) {
      nearMissSlot = candidates[rng.int(candidates.length)];
      const altGlyph = ((goal.glyph + 1 + rng.int(2)) % 3) as GlyphId;
      const t = { color: goal.color, shape: goal.shape, glyph: altGlyph };
      triples.set(nearMissSlot, t);
      used.add(tripleKey(t));
    }
  }

  // Duplicate-pair scene: two devices are observationally identical. No policy can
  // tell them apart, so the only correct behaviour is to refuse.
  let duplicateSlot: number | null = null;
  if (spec.duplicatePair && deviceSlots.size >= 2) {
    const slots = [...deviceSlots].sort((a, b) => a - b);
    const first = slots[0];
    const second = slots[1];
    const t = goal;
    triples.set(first, { ...t });
    triples.set(second, { ...t });
    duplicateSlot = second;
    used.add(tripleKey(t));
  }

  for (let slot = 0; slot < MAX_FIXTURES; slot++) {
    if (triples.has(slot)) continue;
    const available = pool.filter((t) => !used.has(tripleKey(t)));
    const pick = available.length > 0 ? available[rng.int(available.length)] : pool[rng.int(pool.length)];
    triples.set(slot, { ...pick });
    used.add(tripleKey(pick));
  }

  const smudgedSlots = new Set<number>();
  if (spec.duplicatePair) {
    for (const slot of deviceSlots) smudgedSlots.add(slot);
  } else if (spec.nearMiss) {
    // Force the decision to rely on the engraving: both visually identical
    // devices are smudged, so only extra vision separates them.
    if (targetSlot !== null) smudgedSlots.add(targetSlot);
    if (nearMissSlot !== null) smudgedSlots.add(nearMissSlot);
  } else {
    for (const slot of deviceSlots) {
      const smudgeChance = spec.condition === "cluttered" ? 0.5 : 0.35;
      if (rng.next() < smudgeChance) smudgedSlots.add(slot);
    }
  }

  const fixtures: Fixture[] = cells.map((pos, slot) => {
    const kind = deviceSlots.has(slot) ? "device" : "object";
    const triple = triples.get(slot)!;
    return {
      slot,
      kind,
      pos,
      color: triple.color,
      shape: triple.shape,
      glyph: triple.glyph,
      glyphKnown: kind === "object",
      smudged: kind === "device" && smudgedSlots.has(slot),
      inspected: false,
    };
  });

  ensureApproachable(walls, spawn, fixtures);

  let condition: Condition = spec.condition;
  let solvable = true;
  if (spec.noMatch || spec.duplicatePair) solvable = false;
  if (spec.duplicatePair) condition = "unsolvable";
  if (spec.noMatch) condition = "unsolvable";

  const target = spec.noMatch ? null : targetSlot;
  return {
    seed,
    condition,
    solvable,
    gridSize: GRID,
    walls,
    start: spawn,
    goal,
    fixtures,
    hiddenTargetSlot: target,
    hiddenTargetGlyph: spec.noMatch ? null : goal.glyph,
    meta: {
      variant:
        condition === "cluttered" ? 2 : condition === "ambiguous" ? 1 : condition === "unsolvable" ? 3 : 0,
      deviceCount: deviceSlots.size,
      candidateCount: fixtures.filter((f) => f.kind === "device").length,
      visualDecoyObject: fixtures.some((f) => f.kind === "object" && tripleKey(f) === tripleKey(goal)),
      decoySlot: nearMissSlot ?? duplicateSlot,
      smudgedCount: fixtures.filter((f) => f.smudged).length,
    },
  };
}

/** Deterministic public layout signature, used for dataset/version fingerprints. */
export function scenarioSignature(s: Scenario): string {
  return JSON.stringify({
    grid: s.gridSize,
    walls: s.walls.map((w) => (w ? 1 : 0)).join(""),
    start: s.start,
    goal: s.goal,
    condition: s.condition,
    fixtures: s.fixtures.map((f) => [f.pos, f.kind, f.color, f.shape, f.glyph, f.smudged]),
  });
}
