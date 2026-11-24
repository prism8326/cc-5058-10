export type ColorId = 0 | 1 | 2;
export type ShapeId = 0 | 1 | 2;
export type GlyphId = 0 | 1 | 2;
export type FixtureKind = "object" | "device";
export type Condition = "standard" | "ambiguous" | "cluttered" | "unsolvable";

export interface GridPos {
  x: number;
  y: number;
}

export interface Fixture {
  slot: number;
  kind: FixtureKind;
  pos: GridPos;
  color: ColorId;
  shape: ShapeId;
  glyph: GlyphId;
  /** Engraving visible without extra vision (objects only). */
  glyphKnown: boolean;
  /** The mark is smudged; a close inspection only confirms it needs extra vision. */
  smudged: boolean;
  inspected: boolean;
}

export interface GoalCard {
  color: ColorId;
  shape: ShapeId;
  glyph: GlyphId;
}

export interface ScenarioMeta {
  variant: number;
  deviceCount: number;
  candidateCount: number;
  visualDecoyObject: boolean;
  decoySlot: number | null;
  smudgedCount: number;
}

export interface Scenario {
  seed: number;
  condition: Condition;
  solvable: boolean;
  gridSize: number;
  walls: boolean[];
  start: GridPos;
  goal: GoalCard;
  fixtures: Fixture[];
  /** Evaluation-only ground truth. Never passed to policies or the model. */
  hiddenTargetSlot: number | null;
  hiddenTargetGlyph: GlyphId | null;
  meta: ScenarioMeta;
}

export type EndReason =
  | "success"
  | "wrong-target"
  | "budget"
  | "refused"
  | "invalid-limit"
  | "stuck"
  | null;

export interface SlotView {
  slot: number;
  present: boolean;
  kind: FixtureKind;
  color: ColorId;
  shape: ShapeId;
  glyph: GlyphId | null;
  glyphKnown: boolean;
  smudged: boolean;
  inspected: boolean;
  distance: number;
  reachable: boolean;
  adjacent: boolean;
  colorMatch: boolean;
  shapeMatch: boolean;
  glyphMatch: boolean;
  candidate: boolean;
}

export interface ObservationSnapshot {
  seed: number;
  pos: GridPos;
  goal: GoalCard;
  looked: boolean;
  inspected: number[];
  stepsRemaining: number;
  slots: SlotView[];
}

export interface StepRecord {
  index: number;
  action: number;
  legal: boolean;
  reason: string;
  posBefore: GridPos;
  posAfter: GridPos;
  path: GridPos[];
  movementCost: number;
  revealedSlots: number[];
  newInfo: boolean;
  conflict: boolean;
  observation: number[];
  snapshot: ObservationSnapshot;
  legalActions: number[];
  timestamp: number;
}

export interface EpisodeResult {
  seed: number;
  condition: Condition;
  solvable: boolean;
  policyId: string;
  policyKind: PolicyKind;
  success: boolean;
  safeRefusal: boolean;
  endReason: EndReason;
  steps: StepRecord[];
  invalidAttempts: number;
  targetSlot: number | null;
  operatedSlots: number[];
  inspectedSlots: number[];
  looked: boolean;
  revealsUsed: number;
  finalPos: GridPos;
  spentSteps: number;
  pathCost: number;
  stuckReason: string | null;
  conflictSteps: number;
  error: string | null;
  wrongOperates: number;
  wastedSteps: number;
  /** Number of times the arbiter asked the demonstrator (one per low-confidence turn). */
  demonstrationRequests: number;
  /** Per-decision work, measured in the engine rather than estimated. */
  inference: InferenceCost;
  /** Audit trail of the actions actually taken, one per step. */
  decisions: PolicyAudit[];
}

export interface InferenceCost {
  /** Model forward passes (linear or MLP) executed for this episode. */
  forwardPasses: number;
  /** Deterministic feature-inspection operations, summed over passes. */
  featureChecks: number;
  /** Multiply-accumulate operations for the model passes. */
  multiplyAdds: number;
  /** Wall-clock microseconds spent inside policy decision code. */
  decisionMicros: number;
  /** Steps where the arbiter handed control to the demonstrator. */
  demonstratorSteps: number;
  /** Steps where a low-confidence turn instead gathered one more observation. */
  gatherSteps: number;
}

export interface LegalAction {
  action: number;
  legal: boolean;
  reason: string;
}

export interface PolicyFeedback {
  action: number;
  legal: boolean;
  reason: string;
  stepIndex: number;
  endReason: EndReason;
  success: boolean;
  /** The decision that produced `action`, including arbitration provenance. */
  decision?: PolicyDecision;
}

export interface StepObservation {
  vector: number[];
  snapshot: ObservationSnapshot;
  legalActions: LegalAction[];
  stepsRemaining: number;
}

export type PolicyKind = "rule" | "random" | "learned" | "manual" | "demonstrator" | "arbitrated";

export interface DecisionCost {
  forwardPasses: number;
  featureChecks: number;
  multiplyAdds: number;
  /** Wall-clock microseconds for this decision, measured by the engine. */
  micros?: number;
}

export interface PolicyDecision {
  action: number;
  /** Scalar confidence in [0,1] used by arbitration. */
  confidence: number;
  /** Per-action distribution when the policy exposes one (softmax models). */
  probabilities?: number[];
  /** Who actually produced the action, for auditing. */
  source: "learned" | "linear" | "demonstrator" | "fallback" | "manual";
  note?: string;
  /** Work performed to reach this decision; summed by the engine. */
  cost?: DecisionCost;
  /** Arbitration outcome: normal policy step, teacher handover, or evidence gathering. */
  mode?: "policy" | "demonstrator" | "gather";
}

export interface Policy {
  id: string;
  kind: PolicyKind;
  version?: string;
  reset?(seed: number): void;
  select(obs: StepObservation): number;
  /** Optional richer interface; the arbiter and UI prefer it over `select`. */
  decide?(obs: StepObservation): PolicyDecision;
  feedback?(fb: PolicyFeedback): void;
}

/** Subset of the observable state the UI needs to explain a policy action. */
export interface PolicyAudit {
  action: number;
  source: PolicyDecision["source"];
  confidence: number;
  note: string;
  reason: string;
  gatherAction?: number;
}
