import { LOOK_ACTION, MAX_FIXTURES, REFUSE_ACTION, argmax } from "./constants";
import type { EpisodeResult, LegalAction, Policy, SlotView, StepObservation } from "./types";
import type { MLP } from "./model";

const legalSet = (legal: LegalAction[]): Set<number> =>
  new Set(legal.filter((l) => l.legal).map((l) => l.action));

/** Devices still compatible with every observed feature (unknown glyphs included). */
export function plausibleDevices(obs: StepObservation): SlotView[] {
  return obs.snapshot.slots.filter(
    (s) =>
      s.present &&
      s.kind === "device" &&
      s.reachable &&
      s.colorMatch &&
      s.shapeMatch &&
      (!s.glyphKnown || s.glyphMatch),
  );
}

export function confirmedDevices(obs: StepObservation): SlotView[] {
  return obs.snapshot.slots.filter((s) => s.present && s.candidate);
}

/**
 * Two or more devices remain equally consistent with the current observation.
 * This is the conflict the brief asks the demonstrator to acknowledge.
 */
export function hasObservationConflict(obs: StepObservation): boolean {
  return plausibleDevices(obs).length >= 2;
}

/**
 * Stateful rule demonstrator kernel. It receives engine feedback (illegal
 * actions, terminal reasons) and adapts within the episode by avoiding actions
 * that were rejected, which is how failure information flows back to the
 * demonstrator rather than being discarded.
 */
export function makeRuleDemonstrator(): RuleDemonstrator {
  const rejected = new Map<number, string>();
  const feedbackLog: string[] = [];
  let lastFeedback: string | null = null;
  let pendingFeedback: string | null = null;
  return {
    id: "rule-demonstrator",
    kind: "demonstrator",
    reset() {
      rejected.clear();
      lastFeedback = null;
      pendingFeedback = null;
    },
    select(obs) {
      const preferred = demonstratorAction(obs);
      if (!rejected.has(preferred)) return preferred;
      const legal = legalSet(obs.legalActions);
      const fallback = demonstratorFallback(obs, legal, rejected);
      return fallback;
    },
    feedback(fb) {
      if (!fb.legal) {
        rejected.set(fb.action, fb.reason);
        lastFeedback = `动作 #${fb.action} 被引擎拒绝（${fb.reason}），已改走安全备选`;
        pendingFeedback = lastFeedback;
        feedbackLog.push(lastFeedback);
      } else if (fb.endReason === "wrong-target") {
        lastFeedback = "操作错误装置，回合终止；该结果已记录用于诊断";
        pendingFeedback = lastFeedback;
        feedbackLog.push(lastFeedback);
      }
    },
    get feedbackLog() {
      return feedbackLog;
    },
    get lastFeedback() {
      return pendingFeedback;
    },
    consumeFeedback() {
      const message = pendingFeedback;
      pendingFeedback = null;
      return message;
    },
  };
}

export interface RuleDemonstrator extends Policy {
  readonly feedbackLog: string[];
  readonly lastFeedback: string | null;
  /** Returns the latest feedback once, then clears it. */
  consumeFeedback(): string | null;
}

function demonstratorFallback(obs: StepObservation, legal: Set<number>, rejected: Map<number, string>): number {
  // Prefer passive, always-legal termination over repeating a rejected action.
  if (legal.has(REFUSE_ACTION)) return REFUSE_ACTION;
  for (const entry of obs.legalActions) {
    if (entry.legal && !rejected.has(entry.action)) return entry.action;
  }
  return REFUSE_ACTION;
}

/** Shared singleton for CLI pipelines (one episode at a time). */
export const ruleDemonstrator: Policy = makeRuleDemonstrator();

export function demonstratorAction(obs: StepObservation): number {
  const legal = legalSet(obs.legalActions);
  const can = (a: number) => legal.has(a);
  const canLook = !obs.snapshot.looked && can(LOOK_ACTION);
  const plausible = plausibleDevices(obs);
  const confirmed = plausible.filter((s) => s.glyphKnown && s.glyphMatch);
  const unresolved = plausible.filter((s) => !s.glyphKnown);

  if (confirmed.length === 1) return finish(confirmed[0], can);

  if (plausible.length >= 2 || (unresolved.length >= 2 && confirmed.length === 0)) {
    // Multiple candidates and the glyphs can still separate them.
    if (canLook) return LOOK_ACTION;
    if (unresolved.length === 0) {
      // All glyphs known yet more than one full match: genuinely ambiguous.
      return refuseOrInspect(plausible, can);
    }
  }

  if (unresolved.length === 1) {
    const only = unresolved[0];
    if (only.smudged) {
      if (canLook) return LOOK_ACTION;
      return REFUSE_ACTION;
    }
    if (only.adjacent) {
      if (!only.inspected && can(MAX_FIXTURES + only.slot)) return MAX_FIXTURES + only.slot;
      // Inspected but still unreadable: only extra vision can help.
      if (canLook) return LOOK_ACTION;
      return REFUSE_ACTION;
    }
    if (can(only.slot)) return only.slot;
  }

  if (unresolved.length === 0 && confirmed.length === 0) {
    // No device matches the goal's visible colour+shape, and every glyph is
    // already known: the scene offers no valid target.
    return REFUSE_ACTION;
  }

  if (plausible.length === 1) {
    const only = plausible[0];
    if (only.adjacent) {
      if (!only.inspected && can(MAX_FIXTURES + only.slot)) return MAX_FIXTURES + only.slot;
      if (canLook) return LOOK_ACTION;
      return REFUSE_ACTION;
    }
    if (can(only.slot)) return only.slot;
  }

  // Nothing plausible: gather the cheapest decisive evidence once, then refuse.
  if (!obs.snapshot.looked && canLook) {
    const anyDeviceUnread = obs.snapshot.slots.some(
      (s) => s.present && s.kind === "device" && s.reachable && !s.glyphKnown,
    );
    if (anyDeviceUnread) return LOOK_ACTION;
  }
  return REFUSE_ACTION;
}

function finish(target: SlotView, can: (a: number) => boolean): number {
  if (target.adjacent && can(MAX_FIXTURES * 2 + target.slot)) return MAX_FIXTURES * 2 + target.slot;
  if (can(target.slot)) return target.slot;
  return REFUSE_ACTION;
}

function refuseOrInspect(devices: SlotView[], can: (a: number) => boolean): number {
  const adjacent = devices.find((s) => s.adjacent && !s.inspected);
  if (adjacent && can(MAX_FIXTURES + adjacent.slot)) return MAX_FIXTURES + adjacent.slot;
  return REFUSE_ACTION;
}

/** Uniform choice among currently legal actions, seeded per episode. */
export function makeRandomPolicy(policySeed: number): Policy {
  let state = (Math.imul(policySeed, 1664525) + 1013904223) >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return {
    id: "random-legal",
    kind: "random",
    reset: () => {
      state = (Math.imul(policySeed, 1664525) + 1013904223) >>> 0;
    },
    select(obs) {
      const legal = obs.legalActions.filter((l) => l.legal).map((l) => l.action);
      if (legal.length === 0) return REFUSE_ACTION;
      return legal[Math.floor(next() * legal.length)];
    },
  };
}

export interface LearnedPolicy extends Policy {
  lastLogits?: number[];
}

/** Network scores all actions; the engine's legal mask suppresses invalid ones. */
export function makeLearnedPolicy(net: MLP, opts: { id: string; version: string }): LearnedPolicy {
  const policy: LearnedPolicy = {
    id: opts.id,
    kind: "learned",
    version: opts.version,
    select(obs) {
      const logits = net.forward(obs.vector);
      for (let a = 0; a < logits.length; a++) {
        if (a >= obs.legalActions.length || !obs.legalActions[a].legal) logits[a] = -1e9;
      }
      const action = logits.every((v) => v <= -1e8) ? REFUSE_ACTION : argmax(logits);
      policy.lastLogits = logits;
      return action;
    },
  };
  return policy;
}

/** Replays a recorded demonstrator trajectory (used only for dataset checks). */
export function makeReplayPolicy(id: string, actionsBySeed: Map<number, number[]>): Policy {
  let cursor = 0;
  let current: number[] = [];
  return {
    id,
    kind: "demonstrator",
    reset(seed) {
      cursor = 0;
      current = actionsBySeed.get(seed) ?? [];
    },
    select() {
      if (cursor >= current.length) return REFUSE_ACTION;
      return current[cursor++];
    },
  };
}

export type PolicyKind = EpisodeResult["policyKind"];
