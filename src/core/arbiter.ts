import { LOOK_ACTION, MAX_FIXTURES, REFUSE_ACTION } from "./constants";
import type { DecisionCost, Policy, PolicyDecision, SlotView, StepObservation } from "./types";
import { demonstratorAction, makeRuleDemonstrator } from "./policies";
import { macCount, type MLP } from "./model";

/**
 * How the arbiter reacts when the model is not confident enough.
 *  - "defer":  hand the turn to the rule demonstrator (the teacher).
 *  - "gather": spend one observation-gathering action, then re-run the model.
 *  - "both":   keep gathering while evidence remains, otherwise defer.
 */
export type LowConfidenceMode = "defer" | "gather" | "both";

export interface ArbiterConfig {
  /** Minimum top-1 probability over legal actions required to act alone. */
  threshold: number;
  mode: LowConfidenceMode;
  /** Optional cap on deferrals per episode; null means unlimited. */
  maxDeferrals: number | null;
}

export const DEFAULT_ARBITER: ArbiterConfig = {
  threshold: 0.75,
  mode: "both",
  maxDeferrals: null,
};

export interface ArbiterPolicy extends Policy {
  readonly stats: {
    decisions: number;
    deferrals: number;
    gathers: number;
    lowConfidence: number;
  };
  readonly lastDecision: PolicyDecision | null;
}

function legalSet(obs: StepObservation): Set<number> {
  return new Set(obs.legalActions.filter((l) => l.legal).map((l) => l.action));
}

/** Softmax over legal actions only; illegal logits cannot influence confidence. */
export function maskedDistribution(
  logits: number[],
  legal: Set<number>,
): { probs: number[]; best: number; confidence: number } {
  let max = -Infinity;
  for (let a = 0; a < logits.length; a++) if (legal.has(a) && logits[a] > max) max = logits[a];
  const exps = logits.map((v, a) => (legal.has(a) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => (sum > 0 ? v / sum : 0));
  let best = -1;
  let bestP = -1;
  for (let a = 0; a < probs.length; a++) {
    if (legal.has(a) && probs[a] > bestP) {
      bestP = probs[a];
      best = a;
    }
  }
  return { probs, best, confidence: bestP < 0 ? 0 : bestP };
}

/** Top-1 minus runner-up probability over legal actions. */
export function topTwoMargin(probs: number[]): number {
  const ranked = probs.filter((p) => p > 0).sort((a, b) => b - a);
  return (ranked[0] ?? 0) - (ranked[1] ?? 0);
}

/**
 * A legal action that can still add observational evidence. Returns null when
 * nothing observable remains to be gathered.
 */
export function gatherAction(obs: StepObservation, legal: Set<number>, avoid: number | null): number | null {
  const slots: SlotView[] = obs.snapshot.slots;
  const unread = (s: SlotView) => s.present && s.kind === "device" && s.reachable && !s.glyphKnown;
  if (!obs.snapshot.looked && legal.has(LOOK_ACTION) && slots.some(unread)) return LOOK_ACTION;
  for (const s of slots) {
    const inspect = MAX_FIXTURES + s.slot;
    if (s.present && s.kind === "device" && s.adjacent && !s.inspected && legal.has(inspect) && inspect !== avoid) {
      return inspect;
    }
  }
  let bestSlot = -1;
  let bestDist = Infinity;
  for (const s of slots) {
    if (!unread(s)) continue;
    if (s.distance < bestDist && legal.has(s.slot) && s.slot !== avoid) {
      bestDist = s.distance;
      bestSlot = s.slot;
    }
  }
  return bestSlot >= 0 ? bestSlot : null;
}

/**
 * Wraps a softmax model in an arbitration loop. Every emitted action is either
 * the model acting alone (confidence >= threshold), a teacher handover, or an
 * evidence-gathering step; the engine records which and accounts the cost.
 */
export function makeArbiter(
  net: MLP,
  opts: {
    modelKind: "linear" | "mlp";
    modelVersion: string;
    config?: Partial<ArbiterConfig>;
    teacher?: Policy;
  },
): ArbiterPolicy {
  const config: ArbiterConfig = { ...DEFAULT_ARBITER, ...opts.config };
  const teacher = opts.teacher ?? makeRuleDemonstrator();
  const macsPerPass = macCount(net);
  const featureDim = net.sizes[0];
  const stats = { decisions: 0, deferrals: 0, gathers: 0, lowConfidence: 0 };
  let lastDecision: PolicyDecision | null = null;
  let deferrals = 0;
  let lastGatherAction: number | null = null;

  function decide(obs: StepObservation): PolicyDecision {
    stats.decisions += 1;
    const legal = legalSet(obs);
    const logits = net.forward(obs.vector);
    const { probs, best, confidence } = maskedDistribution(logits, legal);
    const cost: DecisionCost = { forwardPasses: 1, featureChecks: featureDim, multiplyAdds: macsPerPass };
    const source = opts.modelKind === "linear" ? "linear" : "learned";

    if (best < 0) {
      lastDecision = { action: REFUSE_ACTION, confidence: 0, source, probabilities: probs, cost, note: "无合法动作，拒绝" };
      return lastDecision;
    }

    if (confidence >= config.threshold) {
      lastDecision = {
        action: best,
        confidence,
        source,
        probabilities: probs,
        cost,
        note: `置信 ${(confidence * 100).toFixed(0)}% ≥ 阈值 ${(config.threshold * 100).toFixed(0)}%，模型独立决定`,
      };
      return lastDecision;
    }

    stats.lowConfidence += 1;

    if (config.mode !== "defer") {
      const gather = gatherAction(obs, legal, lastGatherAction);
      if (gather !== null) {
        stats.gathers += 1;
        lastGatherAction = gather;
        lastDecision = {
          action: gather,
          confidence,
          source,
          probabilities: probs,
          cost,
          mode: "gather",
          note: `置信 ${(confidence * 100).toFixed(0)}% 不足，先补观测（#${gather}）后重新推断`,
        };
        return lastDecision;
      }
    }

    if (config.maxDeferrals === null || deferrals < config.maxDeferrals) {
      const teacherAction = teacher.decide ? teacher.decide(obs).action : demonstratorAction(obs);
      deferrals += 1;
      stats.deferrals += 1;
      lastDecision = {
        action: teacherAction,
        confidence,
        source: "demonstrator",
        probabilities: probs,
        cost,
        mode: "demonstrator",
        note: `置信 ${(confidence * 100).toFixed(0)}% < 阈值 ${(config.threshold * 100).toFixed(0)}%，转交示范者`,
      };
      return lastDecision;
    }

    lastDecision = {
      action: best,
      confidence,
      source,
      probabilities: probs,
      cost,
      note: "置信不足且示范请求已达上限，采用模型首选",
    };
    return lastDecision;
  }

  return {
    id: `arbiter:${opts.modelKind}`,
    kind: "arbitrated",
    version: opts.modelVersion,
    stats,
    get lastDecision() {
      return lastDecision;
    },
    reset(seed: number) {
      stats.decisions = 0;
      stats.deferrals = 0;
      stats.gathers = 0;
      stats.lowConfidence = 0;
      deferrals = 0;
      lastGatherAction = null;
      lastDecision = null;
      teacher.reset?.(seed);
    },
    decide,
    select: (obs) => decide(obs).action,
    feedback(fb) {
      teacher.feedback?.(fb);
    },
  };
}
