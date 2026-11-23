import {
  ACTIONS,
  LOOK_ACTION,
  LOOK_COST,
  MAX_FIXTURES,
  MAX_INVALID_ATTEMPTS,
  MAX_NO_PROGRESS_STREAK,
  REFUSE_ACTION,
  STEP_BUDGET,
  actionKind,
  actionSlot,
} from "./constants";
import { isAdjacent, makeNav, nearestApproach, type NavGrid } from "./nav";
import { buildSlotView, encodeObservation } from "./observation";
import type {
  EpisodeResult,
  GridPos,
  LegalAction,
  Policy,
  Scenario,
  SlotView,
  StepObservation,
  StepRecord,
} from "./types";

interface ExecOutcome {
  legal: boolean;
  reason: string;
  newInfo: boolean;
  path: GridPos[];
  movement: number;
  revealed: number[];
}

export class EpisodeRunner {
  scenario: Scenario;
  nav: NavGrid;
  pos: GridPos;
  stepsRemaining = STEP_BUDGET;
  inspected = new Set<number>();
  revealed = new Set<number>();
  looked = false;
  invalidAttempts = 0;
  spentSteps = 0;
  pathCost = 0;
  revealsUsed = 0;
  operatedSlots: number[] = [];
  steps: StepRecord[] = [];
  noProgress = 0;
  conflictSteps = 0;
  ended = false;
  endReason: EpisodeResult["endReason"] = null;
  wrongOperates = 0;
  wastedSteps = 0;
  stuckReason: string | null = null;

  constructor(scenario: Scenario) {
    this.scenario = scenario;
    this.nav = makeNav(scenario);
    this.pos = { ...scenario.start };
  }

  private distances(): Array<number | null> {
    return this.scenario.fixtures.map((f) => nearestApproach(this.nav, this.pos, f.pos)?.distance ?? null);
  }

  snapshot() {
    const dist = this.distances();
    const slots: SlotView[] = this.scenario.fixtures.map((f, slot) =>
      buildSlotView(this.scenario, slot, {
        distance: dist[slot],
        adjacent: isAdjacent(this.pos, f.pos),
        inspected: this.inspected.has(slot),
        revealed: this.looked || this.revealed.has(slot),
      }),
    );
    return {
      seed: this.scenario.seed,
      pos: { ...this.pos },
      goal: { ...this.scenario.goal },
      looked: this.looked,
      inspected: [...this.inspected].sort((a, b) => a - b),
      stepsRemaining: this.stepsRemaining,
      slots,
    };
  }

  legalActions(): LegalAction[] {
    const dist = this.distances();
    const out: LegalAction[] = [];
    const push = (action: number, legal: boolean, reason: string) => out.push({ action, legal, reason });
    for (let slot = 0; slot < MAX_FIXTURES; slot++) {
      const f = this.scenario.fixtures[slot];
      const d = dist[slot];
      if (d === null) push(slot, false, "目标不可达");
      else if (isAdjacent(this.pos, f.pos)) push(slot, false, "已经相邻，无需接近");
      else if (d > this.stepsRemaining) push(slot, false, `步数不足（需 ${d} 步）`);
      else push(slot, true, "合法");
    }
    for (let slot = 0; slot < MAX_FIXTURES; slot++) {
      const f = this.scenario.fixtures[slot];
      if (f.kind !== "device") push(MAX_FIXTURES + slot, false, "物件没有可检查的标记");
      else if (!isAdjacent(this.pos, f.pos)) push(MAX_FIXTURES + slot, false, "需要先接近");
      else if (this.inspected.has(slot)) push(MAX_FIXTURES + slot, false, "该标记已经检查过");
      else if (this.stepsRemaining < 1) push(MAX_FIXTURES + slot, false, "步数不足");
      else push(MAX_FIXTURES + slot, true, "合法");
    }
    for (let slot = 0; slot < MAX_FIXTURES; slot++) {
      const f = this.scenario.fixtures[slot];
      if (f.kind !== "device") push(MAX_FIXTURES * 2 + slot, false, "物件不是装置，无法操作");
      else if (!isAdjacent(this.pos, f.pos)) push(MAX_FIXTURES * 2 + slot, false, "需要先接近");
      else if (!this.glyphReadable(slot)) push(MAX_FIXTURES * 2 + slot, false, "未读取刻符，不能盲目操作");
      else if (this.stepsRemaining < 1) push(MAX_FIXTURES * 2 + slot, false, "步数不足");
      else push(MAX_FIXTURES * 2 + slot, true, "合法");
    }
    if (this.looked) push(LOOK_ACTION, false, "额外视野已经使用过");
    else if (this.stepsRemaining < LOOK_COST) push(LOOK_ACTION, false, `步数不足（需 ${LOOK_COST} 步）`);
    else push(LOOK_ACTION, true, "合法");
    push(REFUSE_ACTION, true, "合法");
    return out;
  }

  private glyphReadable(slot: number): boolean {
    const f = this.scenario.fixtures[slot];
    if (f.kind === "object") return true;
    return this.looked || this.revealed.has(slot);
  }

  observation(): StepObservation {
    const snapshot = this.snapshot();
    const vector = encodeObservation({
      pos: snapshot.pos,
      goal: snapshot.goal,
      looked: snapshot.looked,
      inspected: snapshot.inspected,
      stepsRemaining: snapshot.stepsRemaining,
      slots: snapshot.slots,
      gridSize: this.scenario.gridSize,
    });
    return { vector, snapshot, legalActions: this.legalActions(), stepsRemaining: this.stepsRemaining };
  }

  private execApproach(slot: number): ExecOutcome {
    const f = this.scenario.fixtures[slot];
    const approach = nearestApproach(this.nav, this.pos, f.pos);
    if (!approach) return { legal: false, reason: "目标不可达", newInfo: false, path: [], movement: 0, revealed: [] };
    this.pos = { ...approach.stand };
    this.pathCost += approach.distance;
    this.spentSteps += approach.distance;
    this.stepsRemaining -= approach.distance;
    return {
      legal: true,
      reason: approach.distance === 0 ? "已经在目标旁" : `沿路径移动 ${approach.distance} 步`,
      newInfo: approach.distance > 0,
      path: approach.path,
      movement: approach.distance,
      revealed: [],
    };
  }

  private execInspect(slot: number): ExecOutcome {
    const f = this.scenario.fixtures[slot];
    this.spentSteps += 1;
    this.stepsRemaining -= 1;
    this.inspected.add(slot);
    const smudged = f.smudged;
    if (!smudged) this.revealed.add(slot);
    return {
      legal: true,
      reason: smudged ? "标记模糊，仅能确认需要额外视野" : `读取到刻符 ${f.glyph}`,
      newInfo: true,
      path: [],
      movement: 0,
      revealed: smudged ? [] : [slot],
    };
  }

  private execOperate(slot: number): ExecOutcome {
    this.spentSteps += 1;
    this.stepsRemaining -= 1;
    this.operatedSlots.push(slot);
    if (slot === this.scenario.hiddenTargetSlot) {
      this.ended = true;
      this.endReason = "success";
      return { legal: true, reason: "操作了正确装置，任务完成", newInfo: true, path: [], movement: 0, revealed: [] };
    }
    this.wrongOperates += 1;
    this.ended = true;
    this.endReason = "wrong-target";
    return { legal: true, reason: "操作了错误装置，任务失败", newInfo: true, path: [], movement: 0, revealed: [] };
  }

  private execLook(): ExecOutcome {
    this.looked = true;
    this.revealsUsed += 1;
    this.spentSteps += LOOK_COST;
    this.stepsRemaining -= LOOK_COST;
    const devices = this.scenario.fixtures
      .filter((f) => f.kind === "device")
      .map((f) => f.slot);
    for (const slot of devices) this.revealed.add(slot);
    return {
      legal: true,
      reason: "请求额外视野：所有装置刻符现已清晰可读",
      newInfo: true,
      path: [],
      movement: 0,
      revealed: devices,
    };
  }

  /** Executes one action. Returns null when the episode has already ended. */
  step(action: number, policy?: Policy): StepRecord | null {
    if (this.ended) return null;
    if (this.stepsRemaining <= 0) {
      this.ended = true;
      this.endReason = "budget";
      this.stuckReason = "步数预算耗尽";
      return null;
    }
    const obs = this.observation();
    const conflict = ambiguousCandidates(obs).length >= 2;
    if (conflict) this.conflictSteps += 1;

    const posBefore = { ...this.pos };
    const revealedBefore = new Set(this.revealed);
    const legalEntry = obs.legalActions[action];
    let outcome: ExecOutcome;

    if (!Number.isInteger(action) || action < 0 || action >= ACTIONS.length) {
      outcome = { legal: false, reason: `动作索引越界: ${action}`, newInfo: false, path: [], movement: 0, revealed: [] };
    } else if (!legalEntry || !legalEntry.legal) {
      outcome = { legal: false, reason: legalEntry?.reason ?? "非法动作", newInfo: false, path: [], movement: 0, revealed: [] };
    } else {
      const kind = actionKind(action);
      const slot = actionSlot(action);
      if (kind === "refuse") {
        this.ended = true;
        this.endReason = "refused";
        outcome = { legal: true, reason: "主动拒绝执行，安全结束", newInfo: false, path: [], movement: 0, revealed: [] };
      } else if (kind === "look") {
        outcome = this.execLook();
      } else if (kind === "approach") {
        outcome = this.execApproach(slot!);
      } else if (kind === "inspect") {
        outcome = this.execInspect(slot!);
      } else {
        outcome = this.execOperate(slot!);
      }
    }

    if (!outcome.legal) {
      this.invalidAttempts += 1;
      if (this.invalidAttempts >= MAX_INVALID_ATTEMPTS) {
        this.ended = true;
        this.endReason = "invalid-limit";
        this.stuckReason = `非法尝试达到 ${MAX_INVALID_ATTEMPTS} 次`;
      }
    } else if (!this.ended) {
      if (outcome.newInfo) this.noProgress = 0;
      else {
        this.noProgress += 1;
        this.wastedSteps += 1;
        if (this.noProgress >= MAX_NO_PROGRESS_STREAK) {
          this.ended = true;
          this.endReason = "stuck";
          this.stuckReason = `连续 ${MAX_NO_PROGRESS_STREAK} 步没有获得新信息`;
        }
      }
      if (this.stepsRemaining <= 0 && !this.ended) {
        this.ended = true;
        this.endReason = "budget";
        this.stuckReason = "步数预算耗尽";
      }
    }

    const record: StepRecord = {
      index: this.steps.length,
      action,
      legal: outcome.legal,
      reason: outcome.reason,
      posBefore,
      posAfter: { ...this.pos },
      path: outcome.path,
      movementCost: outcome.movement,
      revealedSlots: [...this.revealed].filter((s) => !revealedBefore.has(s)),
      newInfo: outcome.newInfo,
      conflict,
      observation: obs.vector,
      snapshot: obs.snapshot,
      legalActions: obs.legalActions.filter((l) => l.legal).map((l) => l.action),
      timestamp: Date.now(),
    };
    this.steps.push(record);

    policy?.feedback?.({
      action,
      legal: outcome.legal,
      reason: outcome.reason,
      stepIndex: record.index,
      endReason: this.endReason,
      success: this.endReason === "success",
    });
    return record;
  }

  result(policyId: string, policyKind: EpisodeResult["policyKind"]): EpisodeResult {
    return {
      seed: this.scenario.seed,
      condition: this.scenario.condition,
      solvable: this.scenario.solvable,
      policyId,
      policyKind,
      success: this.endReason === "success",
      safeRefusal: this.endReason === "refused" && !this.scenario.solvable,
      endReason: this.endReason,
      steps: this.steps,
      invalidAttempts: this.invalidAttempts,
      targetSlot: this.scenario.hiddenTargetSlot,
      operatedSlots: this.operatedSlots,
      inspectedSlots: [...this.inspected].sort((a, b) => a - b),
      looked: this.looked,
      revealsUsed: this.revealsUsed,
      finalPos: { ...this.pos },
      spentSteps: this.spentSteps,
      pathCost: this.pathCost,
      stuckReason: this.stuckReason,
      conflictSteps: this.conflictSteps,
      error: null,
      wrongOperates: this.wrongOperates,
      wastedSteps: this.wastedSteps,
    };
  }
}

export function ambiguousCandidates(obs: StepObservation): number[] {
  return obs.snapshot.slots
    .filter(
      (s) =>
        s.present &&
        s.kind === "device" &&
        s.reachable &&
        s.colorMatch &&
        s.shapeMatch &&
        (!s.glyphKnown || s.glyphMatch),
    )
    .map((s) => s.slot);
}

export interface RunOptions {
  maxSteps?: number;
  onStep?: (record: StepRecord, obs: StepObservation) => void;
}

export function runEpisode(scenario: Scenario, policy: Policy, options: RunOptions = {}): EpisodeResult {
  const runner = new EpisodeRunner(scenario);
  policy.reset?.(scenario.seed);
  const limit = options.maxSteps ?? Number.POSITIVE_INFINITY;
  let obs = runner.observation();
  while (!runner.ended && runner.steps.length < limit) {
    let action: number;
    try {
      action = policy.select(obs);
    } catch (err) {
      runner.ended = true;
      runner.endReason = "stuck";
      runner.stuckReason = `策略异常: ${(err as Error).message}`;
      break;
    }
    const record = runner.step(action, policy);
    if (!record) break;
    options.onStep?.(record, obs);
    obs = runner.observation();
  }
  return runner.result(policy.id, policy.kind);
}
