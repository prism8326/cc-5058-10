import {
  ACTIONS,
  ACTION_COUNT,
  COLORS,
  FEATURE_DIM,
  FEATURE_VERSION,
  GLYPHS,
  GLYPH_SYMBOLS,
  GRID,
  LOOK_ACTION,
  MAX_FIXTURES,
  REFUSE_ACTION,
  SELF_START,
  SHAPES,
  SLOT_FEATURE_FIELDS,
  STEPS_START,
  actionKind,
  seedFamily,
} from "../core/constants";
import { EpisodeRunner, ambiguousCandidates } from "../core/engine";
import { generateScenario, scenarioSpec } from "../core/scenario";
import { MLP, type ModelJSON } from "../core/model";
import {
  makeLearnedPolicy,
  makeRandomPolicy,
  makeRuleDemonstrator,
  demonstratorAction,
  plausibleDevices,
  type RuleDemonstrator,
} from "../core/policies";
import { validateModelFormat } from "../core/validate";
import { makeArbiter, type ArbiterConfig, type ArbiterPolicy, type LowConfidenceMode } from "../core/arbiter";
import type { EpisodeResult, PolicyDecision, Scenario, StepObservation } from "../core/types";
import { Scene3D } from "./scene3d";
import "./styles.css";

type DriverKind = "learned" | "arbiter" | "rule" | "random" | "demo" | "manual";

interface DemoDataset {
  schema: string;
  seeds: number[];
  episodes: Array<{
    seed: number;
    actions: number[];
    success: boolean;
    endReason: string | null;
  }>;
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

const canvas = $<HTMLCanvasElement>("viewport");
const scene3d = new Scene3D(canvas);

let model: MLP | null = null;
let modelMeta: ModelJSON["meta"] | null = null;
let modelError: string | null = null;
let demoData: Map<number, number[]> = new Map();
let driver: DriverKind = "learned";
let scenario: Scenario | null = null;
let runner: EpisodeRunner | null = null;
let running = false;
let busy = false;
let speed = 1.6;
let realtime = true;
let currentSeed = 9000;
let forcedCondition: string = "auto";
let lastObs: StepObservation | null = null;
let discoveredSeeds: number[] = [];
let ruleKernel: RuleDemonstrator | null = null;
let arbiter: ArbiterPolicy | null = null;
let arbiterNet: MLP | null = null;
let arbiterModelKind: "linear" | "mlp" = "linear";
let arbiterMode: LowConfidenceMode = "both";
let arbiterThreshold = 0.8;
let arbiterError: string | null = null;
let arbiterConfigLoaded = false;
let arbiterMetaVersion = "";
let lastDecision: PolicyDecision | null = null;
let arbiterEpisodeSteps = 0;
let arbiterEpisodeGathers = 0;
let arbiterEpisodeDefers = 0;
const arbiterCache = new Map<"linear" | "mlp", MLP | null>();

async function loadModel(): Promise<void> {
  try {
    const res = await fetch("/models/learned-model.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json: ModelJSON = await res.json();
    const fmt = validateModelFormat(json);
    if (!fmt.ok) {
      modelError = fmt.errors.slice(0, 3).join("；");
      return;
    }
    model = MLP.fromJSON(json);
    modelMeta = json.meta;
    const chip = $("model-chip");
    chip.textContent = `模型 ${json.meta.version.slice(-8)} · v${json.meta.featureVersion}`;
    chip.style.borderColor = "rgba(52,211,153,0.45)";
    chip.style.color = "#34d399";
    $("obs-dim").textContent = `obs ${json.meta.featureVersion} · ${json.weights[0][0].length} 维`;
  } catch (err) {
    modelError = (err as Error).message;
    const chip = $("model-chip");
    chip.textContent = "模型加载失败";
    chip.style.borderColor = "rgba(248,113,113,0.5)";
    chip.style.color = "#f87171";
  }
}

async function loadDemos(): Promise<void> {
  try {
    const res = await fetch("/data/train-demonstrations.json", { cache: "no-store" });
    if (!res.ok) return;
    const json: DemoDataset = await res.json();
    for (const ep of json.episodes) demoData.set(ep.seed, ep.actions);
    discoveredSeeds = json.seeds.slice(0, 40);
  } catch {
    // Demonstration replay is optional; the rest of the UI still works.
  }
}

interface ArbiterConfigFile {
  trigger: ArbiterConfig;
  thresholds?: { linear?: Record<string, number>; mlp?: Record<string, number> };
  selectedModels?: { linear?: string; mlp?: string };
}

async function loadArbiterConfig(): Promise<void> {
  try {
    const res = await fetch("/models/arbiter-config.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const cfg: ArbiterConfigFile = await res.json();
    if (cfg.trigger) {
      arbiterThreshold = cfg.trigger.threshold;
      arbiterMode = cfg.trigger.mode;
      $<HTMLInputElement>("arbiter-threshold").value = String(arbiterThreshold);
      $<HTMLSelectElement>("arbiter-mode").value = arbiterMode;
    }
    $("arbiter-status").textContent = "触发条件已载入";
    arbiterConfigLoaded = true;
    renderArbiterStats();
  } catch (err) {
    arbiterError = `仲裁配置未载入：${(err as Error).message}（可运行 npm run demo:experiment 生成）`;
    showArbiterError();
  }
}

async function loadArbiterModel(kind: "linear" | "mlp"): Promise<MLP | null> {
  if (arbiterCache.has(kind)) return arbiterCache.get(kind) ?? null;
  const file = kind === "linear" ? "experiment-linear.json" : "experiment-mlp.json";
  try {
    const res = await fetch(`/models/${file}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    const fmt = validateModelFormat(json, kind === "linear" ? [FEATURE_DIM, ACTION_COUNT] : undefined);
    if (!fmt.ok) throw new Error(fmt.errors.slice(0, 2).join("；"));
    const net = MLP.fromJSON(json);
    arbiterCache.set(kind, net);
    return net;
  } catch (err) {
    arbiterCache.set(kind, null);
    arbiterError = `仲裁模型 ${file} 加载失败：${(err as Error).message}（运行 npm run demo:experiment）`;
    showArbiterError();
    return null;
  }
}

function buildArbiter(): ArbiterPolicy | null {
  if (!arbiterNet) return null;
  const config: ArbiterConfig = { threshold: arbiterThreshold, mode: arbiterMode, maxDeferrals: null };
  const policy = makeArbiter(arbiterNet, {
    modelKind: arbiterModelKind,
    modelVersion: arbiterMetaVersion,
    config,
    teacher: ruleKernel ?? makeRuleDemonstrator(),
  });
  arbiterError = null;
  showArbiterError();
  return policy;
}

function showArbiterError(): void {
  const el = $("arbiter-error");
  if (!arbiterError) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = arbiterError;
}

function renderArbiterStats(): void {
  const container = $("arbiter-stats");
  container.innerHTML = "";
  const stats = arbiter?.stats;
  const entries: Array<[string, string, string]> = [
    ["决策", String(stats?.decisions ?? 0), ""],
    ["补观测", String(stats?.gathers ?? 0), "gather"],
    ["转交示范", String(stats?.deferrals ?? 0), "defer"],
    ["低置信", String(stats?.lowConfidence ?? 0), "bad"],
    ["本回合请求", String(arbiterEpisodeDefers), arbiterEpisodeDefers > 0 ? "defer" : ""],
  ];
  for (const [label, value, cls] of entries) {
    const cell = document.createElement("div");
    cell.className = `arbiter-stat ${cls}`.trim();
    cell.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    container.appendChild(cell);
  }
  const kindLabel = arbiterModelKind === "linear" ? "线性" : "MLP";
  $("arbiter-status").textContent = arbiter
    ? `${kindLabel} · 阈值 ${arbiterThreshold.toFixed(2)} · ${modeLabel(arbiterMode)}`
    : arbiterConfigLoaded
      ? `触发条件已载入 · 阈值 ${arbiterThreshold.toFixed(2)} · 选择“仲裁”驱动`
      : "未启用（选择“仲裁”驱动）";
}

function modeLabel(mode: LowConfidenceMode): string {
  return mode === "defer" ? "转交示范" : mode === "gather" ? "补观测" : "补观测→转交";
}

async function loadEvalReport(): Promise<void> {
  try {
    const res = await fetch("/reports/evaluation.json", { cache: "no-store" });
    if (!res.ok) throw new Error();
    const report = await res.json();
    renderEvalSummary(report);
    await loadModelHistory();
  } catch {
    $("eval-table").textContent = "尚未生成评测报告（运行 npm run evaluate）";
  }
}

function renderEvalSummary(report: {
  modelVersion: string;
  rows?: Array<{
    policy: string;
    group: string;
    episodes: number;
    success: number;
    safeRefusal: number;
    wrongTarget: number;
    invalid: number;
    budget: number;
    stuck: number;
  }>;
  failureBreakdown?: Array<{
    condition: string;
    total: number;
    pass: number;
    wrongTarget: number;
    refused: number;
    budget: number;
    stuck: number;
  }>;
}): void {
  const container = $("eval-table");
  container.innerHTML = "";
  const condNames: Record<string, string> = {
    standard: "标准",
    ambiguous: "歧义",
    cluttered: "拥挤",
    unsolvable: "不可解",
  };
  if (report.failureBreakdown && report.failureBreakdown.length > 0) {
    const block = document.createElement("div");
    block.className = "eval-block";
    const head = document.createElement("div");
    head.className = "eval-block-head";
    head.innerHTML = `<span>学得策略 · 隐藏种子</span><span>${report.modelVersion.slice(-8)}</span>`;
    block.appendChild(head);
    for (const row of report.failureBreakdown) {
      const line = document.createElement("div");
      line.className = "eval-line";
      const rate = Math.round((row.pass / row.total) * 100);
      line.innerHTML = `<span>${condNames[row.condition] ?? row.condition} (${row.total})</span><span>通过 <strong class="${rate === 100 ? "ok" : rate >= 90 ? "" : "warn"}">${rate}%</strong> · 错操作 <strong class="${row.wrongTarget > 0 ? "bad" : "ok"}">${row.wrongTarget}</strong> · 放弃 <strong class="${row.refused > 0 ? "warn" : ""}">${row.refused}</strong></span>`;
      block.appendChild(line);
    }
    container.appendChild(block);
  }
  const block = document.createElement("div");
  block.className = "eval-block";
  const head = document.createElement("div");
  head.className = "eval-block-head";
  head.innerHTML = `<span>对照基线 · 全部隐藏种子</span><span>完整回合</span>`;
  block.appendChild(head);
  const allHidden = report.rows?.filter((r) => r.group === "all-hidden") ?? [];
  const labelFor = (policy: string): string =>
    policy === "rule-demonstrator"
      ? "规则示范"
      : policy === "random-legal"
        ? "随机合法"
        : "学得策略";
  for (const row of allHidden) {
    const line = document.createElement("div");
    line.className = "eval-line";
    const pass = row.success + row.safeRefusal;
    line.innerHTML = `<span>${labelFor(row.policy)}</span><span>通过 <strong class="${pass > row.episodes / 2 ? "ok" : "bad"}">${pass}</strong>/${row.episodes} · 错操作 <strong class="${row.wrongTarget > 0 ? "bad" : "ok"}">${row.wrongTarget}</strong> · 预算 <strong>${row.budget}</strong></span>`;
    block.appendChild(line);
  }
  container.appendChild(block);
}

async function loadModelHistory(): Promise<void> {
  try {
    const res = await fetch("/models/manifest.json", { cache: "no-store" });
    if (!res.ok) return;
    const manifest = await res.json();
    const container = $("eval-table");
    if (!manifest.models || manifest.models.length <= 1) return;
    const block = document.createElement("div");
    block.className = "eval-block";
    const head = document.createElement("div");
    head.className = "eval-block-head";
    head.innerHTML = `<span>模型版本</span><span>${manifest.models.length} 个</span>`;
    block.appendChild(head);
    for (const m of manifest.models.slice(0, 4)) {
      const line = document.createElement("div");
      line.className = "eval-line";
      line.innerHTML = `<span>${m.version.slice(-8)}</span><span>${m.meta.trainSamples} 样本 · ${(m.meta.accuracy * 100).toFixed(1)}%</span>`;
      block.appendChild(line);
    }
    container.appendChild(block);
  } catch {
    // history is optional
  }
}

function conditionSeed(kind: string): number {
  const base: Record<string, number> = {
    standard: 9000,
    ambiguous: 9300,
    cluttered: 9400,
    unsolvable: 9500,
  };
  if (base[kind]) {
    const span = kind === "unsolvable" ? 25 : 100;
    return base[kind] + Math.floor(Math.random() * span);
  }
  return 9000 + Math.floor(Math.random() * 100);
}

function loadScenario(seed: number, conditionOverride?: string): void {
  stopRun();
  currentSeed = seed;
  scenario = generateScenario(seed);
  if (conditionOverride && conditionOverride !== "auto") {
    // Honour the requested family by re-deriving a seed in that family if the
    // current seed produced a different layout kind.
    const want = conditionOverride;
    if (scenario.condition !== want) {
      const seed2 = conditionSeed(want);
      currentSeed = seed2;
      scenario = generateScenario(seed2);
    }
  }
  scene3d.setScenario(scenario);
  scene3d.setCharacterCell(scenario.start);
  runner = new EpisodeRunner(scenario);
  if (ruleKernel) ruleKernel.reset?.(scenario.seed);
  if (driver === "arbiter" && arbiterNet) {
    arbiter = buildArbiter();
    arbiter?.reset?.(scenario.seed);
  }
  lastDecision = null;
  arbiterEpisodeSteps = 0;
  arbiterEpisodeGathers = 0;
  arbiterEpisodeDefers = 0;
  lastObs = runner.observation();
  renderScenarioHeader();
  renderGoal();
  renderObservation(lastObs);
  $("log").innerHTML = "";
  $("end-banner").hidden = true;
  $<HTMLInputElement>("seed-input").value = String(currentSeed);
  updateManualButtons();
  renderArbiterStats();
  scene3d.highlight({
    slots: lastObs.snapshot.slots,
    inspected: [],
    candidateSlot: null,
    adjacentSlot: null,
    revealed: false,
  });
  resetActionPanel();
  const note = $("driver-note");
  note.textContent = driverNote();
  void note;
}

function driverNote(): string {
  switch (driver) {
    case "learned":
      return "小模型只接收 139 维可观测特征，输出动作用引擎合法性掩码过滤。不能读取隐藏目标槽。";
    case "arbiter":
      return "仲裁：模型先给出合法动作分布；置信低于阈值时，先补观测或把该回合转交示范者。来源、置信与推理成本写入日志。";
    case "rule":
      return "规则示范策略读取同一观测向量，按颜色/形状缩小候选，再用刻符与视野消解歧义，必要时拒绝。";
    case "random":
      return "在每个状态从当前合法动作中均匀随机选择，作为下限基线。";
    case "demo":
      return "回放离线采集的示范动作序列；若当前种子不在示范数据集中，将回退为规则示范。";
    case "manual":
      return "由你直接驱动角色。非法动作会被拒绝并把原因反馈到日志。";
  }
}

function renderScenarioHeader(): void {
  if (!scenario) return;
  $("seed-chip").textContent = `场景 #${scenario.seed}`;
  const family = seedFamily(scenario.seed);
  const familyLabels: Record<string, string> = {
    train: "训练种子",
    dev: "开发种子",
    validation: "验证种子",
    novel: "新布局种子",
    "hidden-eval": "隐藏评测种子",
    manual: "手工种子",
  };
  const condLabels: Record<string, string> = {
    standard: "标准",
    ambiguous: "歧义",
    cluttered: "拥挤",
    unsolvable: "不可解",
  };
  $("family-chip").textContent = `${familyLabels[family] ?? family} · ${condLabels[scenario.condition]}`;
  $("family-chip").style.color = family === "hidden-eval" ? "#fbbf24" : family === "novel" ? "#7dd3fc" : "#8fa1c4";
}

function renderGoal(): void {
  if (!scenario) return;
  $("goal-glyph").textContent = GLYPH_SYMBOLS[scenario.goal.glyph];
  $("goal-color").textContent = COLORS[scenario.goal.color];
  $("goal-shape").textContent = SHAPES[scenario.goal.shape];
}

function policyForCurrent(): { select: (obs: StepObservation) => number; id: string } {
  if (driver === "learned") {
    if (model) return makeLearnedPolicy(model, { id: `learned:${modelMeta?.version ?? "local"}`, version: modelMeta?.version ?? "local" });
    // Missing model must never silently fall back to the demonstrator.
    return { id: "missing-model", select: () => -1 };
  }
  if (driver === "arbiter") {
    if (!arbiter) arbiter = buildArbiter();
    if (!arbiter) return { id: "missing-arbiter", select: () => -1 };
    return arbiter;
  }
  if (driver === "rule") {
    if (!ruleKernel) ruleKernel = makeRuleDemonstrator();
    return ruleKernel;
  }
  if (driver === "random") {
    const p = makeRandomPolicy(currentSeed);
    return { id: p.id, select: p.select };
  }
  if (driver === "demo") {
    const actions = demoData.get(currentSeed);
    if (actions) {
      let cursor = 0;
      return {
        id: "replay-demonstration",
        select: () => (cursor < actions.length ? actions[cursor++] : REFUSE_ACTION),
      };
    }
    return { id: "rule-fallback", select: (obs) => demonstratorAction(obs) };
  }
  return { id: "manual", select: () => REFUSE_ACTION };
}

function chooseAction(): number {
  if (!runner || !scenario) return -1;
  const obs = runner.observation();
  if (driver === "manual") return -1;
  const policy = policyForCurrent();
  if (policy.id === "missing-model" || policy.id === "missing-arbiter") return -1;
  if (driver === "arbiter" && arbiter) {
    const decision = arbiter.decide!(obs);
    lastDecision = decision;
    renderArbiterDecision(decision, obs, "arbiter");
    return decision.action;
  }
  const action = policy.select(obs);
  // Surface model probabilities for the action panel.
  if (driver === "learned" && model) {
    const logits = model.forward(obs.vector);
    for (let a = 0; a < logits.length; a++) {
      if (a >= obs.legalActions.length || !obs.legalActions[a].legal) logits[a] = -1e9;
    }
    renderActionProbabilities(logits, action, obs);
  }
  return action;
}

async function executeAction(action: number): Promise<void> {
  if (!runner || busy) return;
  const obs = runner.observation();
  lastObs = obs;
  const decision = driver === "arbiter" ? lastDecision ?? undefined : undefined;
  const record = runner.step(
    action,
    driver === "rule" ? (ruleKernel ?? undefined) : driver === "arbiter" ? (arbiter ?? undefined) : undefined,
    decision,
  );
  if (!record) {
    finishIfEnded();
    return;
  }
  if (driver === "rule" && ruleKernel) {
    const feedback = ruleKernel.consumeFeedback?.();
    if (feedback) appendLog(record.index, -1, feedback, false, false, "rule", "refuse");
  }
  renderActionNow(record.action, record.reason, record.legal);
  renderObservation(obs);
  appendLog(record.index, record.action, record.reason, record.legal, record.conflict, driver, undefined, decision);
  if (record.action === LOOK_ACTION) scene3d.playLookSweep();

  const highlight = {
    slots: runner.snapshot().slots,
    inspected: [...runner.inspected],
    candidateSlot: ambiguousCandidates(obs)[0] ?? null,
    adjacentSlot: obs.snapshot.slots.find((s) => s.adjacent)?.slot ?? null,
    revealed: runner.looked,
  };

  if (realtime && record.path.length > 0) {
    busy = true;
    await animateAtSpeed(record.path);
    busy = false;
  } else {
    scene3d.setCharacterCell(record.posAfter);
  }
  scene3d.highlight(highlight);
  updateStats();
  updateManualButtons();
  renderArbiterStats();
  finishIfEnded();
}

function animateAtSpeed(path: Array<{ x: number; y: number }>): Promise<void> {
  return scene3d.animateAlong(path, speed);
}

function updateStats(): void {
  if (!runner) return;
  $("steps-left").textContent = String(runner.stepsRemaining);
  $("look-state").textContent = runner.looked ? "已使用" : "未使用";
  $("look-state").style.color = runner.looked ? "#fbbf24" : "#7dd3fc";
  $("conflict-count").textContent = String(runner.conflictSteps);
}

function finishIfEnded(): void {
  if (!runner || !runner.ended) return;
  const result: EpisodeResult = runner.result(driver, driverKind());
  const banner = $("end-banner");
  banner.hidden = false;
  banner.className = "banner";
  const targetNote =
    result.targetSlot === null ? "" : `（正确槽位 #${result.targetSlot}）`;
  const arbiterNote =
    driver === "arbiter"
      ? ` · 示范请求 ${result.demonstrationRequests} 次 · 前向 ${result.inference.forwardPasses} 次`
      : "";
  if (result.success) {
    banner.classList.add("success");
    banner.textContent = `任务完成 ✓ 操作正确装置 #${result.operatedSlots.at(-1)}${targetNote}${arbiterNote}`;
  } else if (result.safeRefusal) {
    banner.classList.add("refuse");
    banner.textContent = `安全拒绝 ✓ 场景不可解，代理主动终止${targetNote}${arbiterNote}`;
  } else if (result.endReason === "refused") {
    banner.classList.add("refuse");
    banner.textContent = `代理拒绝执行（场景其实可解）${targetNote}${arbiterNote}`;
  } else {
    banner.classList.add("failure");
    const reasons: Record<string, string> = {
      "wrong-target": `操作了错误装置`,
      budget: "步数预算耗尽",
      "invalid-limit": "非法尝试过多",
      stuck: "代理卡住（无新信息）",
    };
    banner.textContent = `失败：${reasons[result.endReason ?? ""] ?? result.endReason}${targetNote}${arbiterNote}`;
  }
  appendLog(
    result.steps.length,
    -1,
    result.stuckReason ?? (result.success ? "回合结束：成功" : `回合结束：${result.endReason}`),
    false,
    false,
    driver,
    result.success ? "success" : "invalid",
  );
  if (driver === "arbiter") {
    $("arbiter-status").textContent = `完成 · 请求 ${result.demonstrationRequests} · 前向 ${result.inference.forwardPasses}`;
  }
  stopRun();
}

function driverKind(): EpisodeResult["policyKind"] {
  if (driver === "learned") return "learned";
  if (driver === "arbiter") return "arbitrated";
  if (driver === "rule") return "demonstrator";
  if (driver === "random") return "random";
  return "manual";
}

function resetActionPanel(): void {
  $("action-label").textContent = "—";
  $("action-reason").textContent = "等待开始";
  $("action-probs").innerHTML = "";
  updateStats();
}

function renderActionNow(action: number, reason: string, legal: boolean): void {
  const spec = ACTIONS[action];
  $("action-label").textContent = spec ? spec.label : `动作 ${action}`;
  $("action-label").style.color = legal ? "#7dd3fc" : "#f87171";
  $("action-reason").textContent = reason;
  $("action-timing").textContent =
    driver === "learned"
      ? "模型输出"
      : driver === "arbiter" && lastDecision
        ? decisionSourceLabel(lastDecision)
        : driver === "rule"
          ? "规则输出"
          : driver === "demo"
            ? "示范回放"
            : "手动";
}

function decisionSourceLabel(decision: PolicyDecision): string {
  if (decision.mode === "gather") return "仲裁 · 补观测";
  if (decision.mode === "demonstrator") return "仲裁 · 转交示范";
  return "仲裁 · 模型独立";
}

function sourceClass(decision: PolicyDecision | null | undefined): "model" | "gather" | "defer" | null {
  if (!decision) return null;
  if (decision.mode === "gather") return "gather";
  if (decision.mode === "demonstrator") return "defer";
  return "model";
}

/** Renders arbitration provenance, confidence and measured cost for the last turn. */
function renderArbiterDecision(decision: PolicyDecision, obs: StepObservation, kind: string): void {
  const spec = ACTIONS[decision.action];
  $("action-label").textContent = spec ? spec.label : `动作 ${decision.action}`;
  $("action-label").style.color = "#7dd3fc";
  $("action-reason").textContent = decision.note ?? "";
  $("action-timing").textContent = decisionSourceLabel(decision);
  if (decision.mode === "demonstrator") arbiterEpisodeDefers += 1;
  if (decision.mode === "gather") arbiterEpisodeGathers += 1;
  arbiterEpisodeSteps += 1;

  const container = $("action-probs");
  container.innerHTML = "";
  if (decision.probabilities) {
    const ranked = decision.probabilities
      .map((p, a) => ({ p, a }))
      .filter((x) => x.p > 0.0005)
      .sort((x, y) => y.p - x.p)
      .slice(0, 6);
    for (const { p, a } of ranked) {
      const row = document.createElement("div");
      row.className = `prob-row${a === decision.action ? " top" : ""}`;
      row.innerHTML = `<span>${ACTIONS[a]?.label ?? a}</span><div class="prob-bar"><div class="prob-fill" style="width:${(p * 100).toFixed(1)}%"></div></div><span>${(p * 100).toFixed(0)}%</span>`;
      container.appendChild(row);
    }
  }
  const trust = document.createElement("div");
  trust.className = "trust-bar";
  trust.innerHTML = `<div style="width:${(decision.confidence * 100).toFixed(1)}%"></div>`;
  container.appendChild(trust);
  const meta = document.createElement("div");
  meta.className = "obs-zero";
  const cost = decision.cost;
  meta.textContent = `置信 ${(decision.confidence * 100).toFixed(0)}% · 阈值 ${(arbiterThreshold * 100).toFixed(0)}% · 前向 ${cost?.forwardPasses ?? 0} · MAC ${cost?.multiplyAdds ?? 0}`;
  container.appendChild(meta);
  void obs;
  void kind;
}

function renderActionProbabilities(logits: number[], chosen: number, obs: StepObservation): void {
  const max = Math.max(...logits.filter((v) => v > -1e8));
  const exps = logits.map((v) => (v <= -1e8 ? 0 : Math.exp(v - max)));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  const probs = exps.map((v) => v / sum);
  const ranked = probs
    .map((p, a) => ({ p, a }))
    .filter((x) => logits[x.a] > -1e8)
    .sort((x, y) => y.p - x.p)
    .slice(0, 6);
  const container = $("action-probs");
  container.innerHTML = "";
  for (const { p, a } of ranked) {
    const legal = obs.legalActions[a]?.legal;
    const row = document.createElement("div");
    row.className = `prob-row${a === chosen ? " top" : ""}`;
    row.innerHTML = `<span>${ACTIONS[a]?.label ?? a}</span><div class="prob-bar"><div class="prob-fill" style="width:${(p * 100).toFixed(1)}%"></div></div><span>${(p * 100).toFixed(0)}%</span>`;
    if (!legal) row.style.opacity = "0.35";
    container.appendChild(row);
  }
}

function renderObservation(obs: StepObservation): void {
  const container = $("obs-groups");
  container.innerHTML = "";
  const slots = obs.snapshot.slots.filter((s) => s.present);
  for (const s of slots) {
    const group = document.createElement("div");
    group.className = "obs-group";
    const title = document.createElement("div");
    title.className = "obs-group-title";
    const glyphText = s.glyphKnown && s.glyph !== null ? `${GLYPH_SYMBOLS[s.glyph]}` : s.smudged ? "模糊" : "未读";
    title.innerHTML = `<span>槽 #${s.slot} · ${s.kind === "device" ? "装置" : "物件"}</span><span>${COLORS[s.color]} ${SHAPES[s.shape]} 刻符${glyphText} · 距离${s.distance}</span>`;
    group.appendChild(title);
    const flags = document.createElement("div");
    flags.className = "obs-flags";
    const add = (label: string, cls: string) => {
      const span = document.createElement("span");
      span.className = `obs-flag ${cls}`;
      span.textContent = label;
      flags.appendChild(span);
    };
    if (s.colorMatch) add("颜色相符", "match");
    if (s.shapeMatch) add("形状相符", "match");
    if (s.glyphKnown) add(s.glyphMatch ? "刻符相符" : "刻符不符", s.glyphMatch ? "match" : "on");
    else add(s.smudged ? "刻符模糊" : "刻符未读", "unknown");
    if (s.adjacent) add("相邻", "on");
    if (s.inspected) add("已检查", "on");
    if (s.candidate) add("完整候选", "match");
    if (!s.reachable) add("不可达", "unknown");
    group.appendChild(flags);
    container.appendChild(group);
  }

  const globals = document.createElement("div");
  globals.className = "obs-group";
  const gTitle = document.createElement("div");
  gTitle.className = "obs-group-title";
  const plausible = plausibleDevices(obs).length;
  gTitle.innerHTML = `<span>全局观测</span><span>候选装置 ${plausible} 个</span>`;
  globals.appendChild(gTitle);
  const gFlags = document.createElement("div");
  gFlags.className = "obs-flags";
  const gf = (label: string, cls: string) => {
    const span = document.createElement("span");
    span.className = `obs-flag ${cls}`;
    span.textContent = label;
    gFlags.appendChild(span);
  };
  gf(`坐标 (${obs.snapshot.pos.x},${obs.snapshot.pos.y})`, "unknown");
  gf(`剩余 ${obs.stepsRemaining} 步`, "unknown");
  gf(obs.snapshot.looked ? "额外视野已用" : "额外视野可用", obs.snapshot.looked ? "on" : "unknown");
  if (plausible >= 2) gf("观测存在冲突", "match");
  if (obs.snapshot.inspected.length > 0) gf(`已检查 ${obs.snapshot.inspected.map((s) => `#${s}`).join(" ")}`, "on");
  globals.appendChild(gFlags);
  container.appendChild(globals);

  const raw = document.createElement("div");
  raw.className = "obs-group";
  const rTitle = document.createElement("div");
  rTitle.className = "obs-group-title";
  rTitle.innerHTML = `<span>编码前 24 维</span><span>共 ${obs.vector.length} 维</span>`;
  raw.appendChild(rTitle);
  const rawFlags = document.createElement("div");
  rawFlags.className = "obs-flags";
  obs.vector.slice(0, 24).forEach((v, i) => {
    if (v === 0) return;
    const span = document.createElement("span");
    span.className = "obs-flag on";
    span.textContent = `${i}:${v.toFixed(2)}`;
    rawFlags.appendChild(span);
  });
  if (rawFlags.children.length === 0) {
    const span = document.createElement("span");
    span.className = "obs-zero";
    span.textContent = "前 24 维全为 0";
    rawFlags.appendChild(span);
  }
  raw.appendChild(rawFlags);
  container.appendChild(raw);

  void SLOT_FEATURE_FIELDS;
  void SELF_START;
  void STEPS_START;
  void GRID;
  void MAX_FIXTURES;
  void ACTION_COUNT;
  void FEATURE_VERSION;
  void GLYPHS;
}

function appendLog(
  index: number,
  action: number,
  reason: string,
  legal: boolean,
  conflict: boolean,
  driverKindValue: string,
  forceClass?: string,
  decision?: PolicyDecision | null,
): void {
  const list = $("log");
  const li = document.createElement("li");
  const cls = forceClass ?? (!legal ? "invalid" : action === REFUSE_ACTION ? "refuse" : driverKindValue === "learned" ? "learned" : "");
  if (cls) li.classList.add(cls);
  if (cls === "success") li.classList.add("success");
  const label = action >= 0 ? ACTIONS[action]?.label ?? `动作 ${action}` : "系统";
  const sourceCls = sourceClass(decision);
  const source =
    sourceCls && decision
      ? `<span class="log-source ${sourceCls}">${decisionSourceLabel(decision)} ${(decision.confidence * 100).toFixed(0)}%</span>`
      : "";
  li.innerHTML = `<span class="log-idx">${String(index).padStart(2, "0")}</span><span class="log-action">${label}</span> — ${reason}${source}${conflict ? '<span class="log-conflict">⚠ 观测冲突</span>' : ""}`;
  list.appendChild(li);
  list.scrollTop = list.scrollHeight;
}

function updateManualButtons(): void {
  const panel = $("manual-panel");
  panel.hidden = driver !== "manual";
  if (driver !== "manual" || !runner) return;
  const legal = runner.legalActions();
  const container = $("manual-actions");
  container.innerHTML = "";
  for (const spec of ACTIONS) {
    const info = legal[spec.index];
    const btn = document.createElement("button");
    btn.textContent = spec.label;
    btn.disabled = !info?.legal;
    btn.title = info?.reason ?? "";
    if (spec.kind === "look") btn.className = "look";
    if (spec.kind === "refuse") btn.className = "refuse";
    btn.addEventListener("click", () => {
      if (!info?.legal) return;
      void executeAction(spec.index);
    });
    container.appendChild(btn);
  }
}

function stopRun(): void {
  running = false;
  const btn = $<HTMLButtonElement>("btn-play");
  btn.textContent = "▶ 代理驱动";
}

async function runLoop(maxSteps = Infinity): Promise<void> {
  if (!runner) return;
  let steps = 0;
  while (runner && !runner.ended && steps < maxSteps) {
    if (driver === "manual") break;
    if (modelError && driver === "learned") {
      appendLog(steps, -1, `模型不可用：${modelError}`, false, false, driver, "invalid");
      break;
    }
    if (driver === "arbiter" && !arbiterNet) {
      const message = arbiterError ?? "仲裁模型不可用";
      appendLog(steps, -1, message, false, false, driver, "invalid");
      arbiterError = message;
      showArbiterError();
      break;
    }
    let action: number;
    try {
      action = chooseAction();
    } catch (err) {
      const message = `仲裁决策异常：${(err as Error).message}`;
      appendLog(steps, -1, message, false, false, driver, "invalid");
      arbiterError = message;
      showArbiterError();
      break;
    }
    if (action < 0) break;
    await executeAction(action);
    steps++;
    if (realtime) await delay(260 / speed);
    if (!running && maxSteps === Infinity) break;
  }
}

async function delay(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function bindUi(): void {
  for (const btn of $("driver-switch").querySelectorAll<HTMLButtonElement>("button")) {
    btn.addEventListener("click", () => {
      driver = btn.dataset.driver as DriverKind;
      for (const b of $("driver-switch").querySelectorAll("button")) b.classList.toggle("active", b === btn);
      stopRun();
      void activateDriver();
      $("driver-note").textContent = driverNote();
      updateManualButtons();
      renderArbiterStats();
    });
  }

  $("arbiter-model").addEventListener("change", () => {
    void selectArbiterModel($<HTMLSelectElement>("arbiter-model").value as "linear" | "mlp");
  });

  $("arbiter-mode").addEventListener("change", () => {
    arbiterMode = $<HTMLSelectElement>("arbiter-mode").value as LowConfidenceMode;
    if (driver === "arbiter") {
      arbiter = buildArbiter();
      arbiter?.reset?.(currentSeed);
      loadScenario(currentSeed);
    }
    renderArbiterStats();
  });

  $("arbiter-threshold").addEventListener("change", () => {
    const value = Number($<HTMLInputElement>("arbiter-threshold").value);
    if (Number.isFinite(value) && value > 0 && value < 1) {
      arbiterThreshold = value;
      if (driver === "arbiter") {
        arbiter = buildArbiter();
        arbiter?.reset?.(currentSeed);
        loadScenario(currentSeed);
      }
      renderArbiterStats();
    }
  });

  $("btn-play").addEventListener("click", () => {
    if (running) {
      stopRun();
      return;
    }
    if (!runner || runner.ended) loadScenario(currentSeed);
    running = true;
    const btn = $<HTMLButtonElement>("btn-play");
    btn.textContent = "⏸ 暂停";
    void runLoop().finally(() => {
      running = false;
      btn.textContent = "▶ 代理驱动";
    });
  });

  $("btn-step").addEventListener("click", () => {
    if (busy) return;
    if (!runner || runner.ended) return;
    if (driver === "manual") {
      updateManualButtons();
      return;
    }
    if (modelError && driver === "learned") {
      appendLog(-1, -1, `模型不可用：${modelError}`, false, false, driver, "invalid");
      return;
    }
    if (driver === "arbiter" && !arbiterNet) {
      const message = arbiterError ?? "仲裁模型不可用";
      appendLog(-1, -1, message, false, false, driver, "invalid");
      arbiterError = message;
      showArbiterError();
      return;
    }
    const action = chooseAction();
    if (action >= 0) void executeAction(action);
  });

  $("btn-reset").addEventListener("click", () => loadScenario(currentSeed));

  $("btn-demo").addEventListener("click", () => {
    driver = "demo";
    for (const b of $("driver-switch").querySelectorAll("button")) {
      b.classList.toggle("active", (b as HTMLElement).dataset.driver === "demo");
    }
    $("driver-note").textContent = driverNote();
    loadScenario(currentSeed);
    const has = demoData.has(currentSeed);
    if (!has) {
      appendLog(-1, -1, `种子 #${currentSeed} 不在示范数据集中，回退为规则示范… 可试训练种子 1000–1279 / 1600–1619`, false, false, "demo", "refuse");
      if (discoveredSeeds.length > 0) {
        const pick = discoveredSeeds[Math.floor(Math.random() * discoveredSeeds.length)];
        loadScenario(pick);
        $("driver-note").textContent = "已切换到示范数据集内的种子，点“代理驱动”即可回放。";
      }
    }
    updateManualButtons();
  });

  $("btn-random-scene").addEventListener("click", () => {
    const pool = ["standard", "ambiguous", "cluttered"];
    const kind = pool[Math.floor(Math.random() * pool.length)];
    loadScenario(conditionSeed(kind));
  });

  $("btn-load-seed").addEventListener("click", () => {
    const val = Number($<HTMLInputElement>("seed-input").value);
    if (Number.isFinite(val) && val > 0) loadScenario(Math.floor(val));
  });

  $("seed-input").addEventListener("keydown", (e) => {
    if ((e as KeyboardEvent).key === "Enter") $<HTMLButtonElement>("btn-load-seed").click();
  });

  $("condition-select").addEventListener("change", () => {
    forcedCondition = $<HTMLSelectElement>("condition-select").value;
    if (forcedCondition !== "auto") loadScenario(conditionSeed(forcedCondition), "auto");
  });

  $("chk-realtime").addEventListener("change", () => {
    realtime = $<HTMLInputElement>("chk-realtime").checked;
  });

  const speedInput = $<HTMLInputElement>("speed-input");
  speedInput.addEventListener("input", () => {
    speed = Number(speedInput.value);
    $("speed-value").textContent = `${speed.toFixed(1)}×`;
    scene3d.animationSpeed = speed;
  });
}

/** Switches to the selected driver, loading whatever model that driver needs. */
async function activateDriver(): Promise<void> {
  if (driver === "arbiter") {
    arbiterNet = await loadArbiterModel(arbiterModelKind);
    arbiterMetaVersion = arbiterModelKind === "linear" ? "linear-experiment" : "mlp-experiment";
    arbiter = buildArbiter();
  }
  if (driver === "rule" && !ruleKernel) ruleKernel = makeRuleDemonstrator();
  if (runner && !runner.ended) {
    // Restart from the current scenario so the driver owns a clean episode.
    loadScenario(currentSeed);
  } else if (runner && runner.ended) {
    loadScenario(currentSeed);
  }
  renderArbiterStats();
}

async function selectArbiterModel(kind: "linear" | "mlp"): Promise<void> {
  arbiterModelKind = kind;
  if (driver === "arbiter") {
    arbiterNet = await loadArbiterModel(kind);
    arbiterMetaVersion = kind === "linear" ? "linear-experiment" : "mlp-experiment";
    arbiter = buildArbiter();
    loadScenario(currentSeed);
  }
  renderArbiterStats();
}

async function boot(): Promise<void> {
  bindUi();
  await Promise.all([loadModel(), loadDemos()]);
  await loadArbiterConfig();
  void loadEvalReport();
  loadScenario(currentSeed, forcedCondition === "auto" ? undefined : forcedCondition);
  $("action-timing").textContent = "";
  void scenarioSpec;
  void Scene3D;
}

void boot();
