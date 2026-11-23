import type { GoalCard, GridPos, Scenario, SlotView } from "./types";
import {
  FEATURE_DIM,
  GOAL_COLOR_START,
  GOAL_GLYPH_START,
  GOAL_SHAPE_START,
  INSPECTED_START,
  LOOKED_START,
  MAX_FIXTURES,
  PER_SLOT_FEATURES,
  SELF_START,
  STEP_BUDGET,
  STEPS_START,
} from "./constants";

export const oneHot = (target: number, size = 3): number[] => {
  const out = new Array(size).fill(0);
  if (target >= 0 && target < size) out[target] = 1;
  return out;
};

export interface RawObservation {
  pos: GridPos;
  goal: GoalCard;
  looked: boolean;
  inspected: number[];
  stepsRemaining: number;
  slots: SlotView[];
  gridSize: number;
}

/** Encodes only information the character can actually observe. */
export function encodeObservation(obs: RawObservation): number[] {
  const vec = new Array(FEATURE_DIM).fill(0);
  const span = Math.max(1, obs.gridSize - 1);

  for (const s of obs.slots) {
    const b = s.slot * PER_SLOT_FEATURES;
    const c = oneHot(s.color);
    const sh = oneHot(s.shape);
    const g = oneHot(s.glyph ?? -1);
    vec[b + 0] = s.present ? 1 : 0;
    vec[b + 1] = s.kind === "device" ? 1 : 0;
    vec[b + 2] = c[0];
    vec[b + 3] = c[1];
    vec[b + 4] = c[2];
    vec[b + 5] = sh[0];
    vec[b + 6] = sh[1];
    vec[b + 7] = sh[2];
    vec[b + 8] = g[0];
    vec[b + 9] = g[1];
    vec[b + 10] = g[2];
    vec[b + 11] = s.glyphKnown ? 1 : 0;
    vec[b + 12] = s.smudged ? 1 : 0;
    vec[b + 13] = s.reachable ? Math.min(s.distance, 24) / 24 : 1;
    vec[b + 14] = s.reachable ? 1 : 0;
    vec[b + 15] = s.adjacent ? 1 : 0;
    vec[b + 16] = s.colorMatch ? 1 : 0;
    vec[b + 17] = s.shapeMatch ? 1 : 0;
    vec[b + 18] = s.glyphMatch ? 1 : 0;
    vec[b + 19] = s.candidate ? 1 : 0;
  }

  const gc = oneHot(obs.goal.color);
  const gs = oneHot(obs.goal.shape);
  const gg = oneHot(obs.goal.glyph);
  for (let i = 0; i < 3; i++) {
    vec[GOAL_COLOR_START + i] = gc[i];
    vec[GOAL_SHAPE_START + i] = gs[i];
    vec[GOAL_GLYPH_START + i] = gg[i];
  }
  const inspected = new Set(obs.inspected);
  for (let i = 0; i < MAX_FIXTURES; i++) vec[INSPECTED_START + i] = inspected.has(i) ? 1 : 0;
  vec[LOOKED_START] = obs.looked ? 1 : 0;
  vec[SELF_START] = obs.pos.x / span;
  vec[SELF_START + 1] = obs.pos.y / span;
  vec[STEPS_START] = Math.max(0, Math.min(obs.stepsRemaining, STEP_BUDGET)) / STEP_BUDGET;

  return vec;
}

export interface SlotFacts {
  distance: number | null;
  adjacent: boolean;
  inspected: boolean;
  revealed: boolean;
}

/**
 * Builds the per-slot observable view. Devices conceal their engraving until
 * extra vision reveals it (or a close inspection reads a non-smudged mark);
 * objects are always readable.
 */
export function buildSlotView(scenario: Scenario, slot: number, facts: SlotFacts): SlotView {
  const f = scenario.fixtures[slot];
  if (!f) {
    return {
      slot,
      present: false,
      kind: "object",
      color: 0,
      shape: 0,
      glyph: null,
      glyphKnown: false,
      smudged: false,
      inspected: false,
      distance: 99,
      reachable: false,
      adjacent: false,
      colorMatch: false,
      shapeMatch: false,
      glyphMatch: false,
      candidate: false,
    };
  }
  const glyphVisible = f.kind === "object" || facts.revealed;
  const glyph = glyphVisible ? f.glyph : null;
  const colorMatch = f.color === scenario.goal.color;
  const shapeMatch = f.shape === scenario.goal.shape;
  const glyphMatch = glyph !== null && glyph === scenario.goal.glyph;
  return {
    slot,
    present: true,
    kind: f.kind,
    color: f.color,
    shape: f.shape,
    glyph,
    glyphKnown: glyph !== null,
    smudged: f.smudged && !glyphVisible,
    inspected: facts.inspected,
    distance: facts.distance ?? 99,
    reachable: facts.distance !== null,
    adjacent: facts.adjacent,
    colorMatch,
    shapeMatch,
    glyphMatch,
    candidate: f.kind === "device" && colorMatch && shapeMatch && glyphMatch,
  };
}
