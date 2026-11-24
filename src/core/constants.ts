export const ENGINE_VERSION = "1.0.0";
export const FEATURE_VERSION = "obs-v2";
export const ACTION_VERSION = "act-v2";
export const DATASET_VERSION = "demo-v2";
export const CONFIDENCE_VERSION = "conf-v1";
export const ARBITRATION_VERSION = "arb-v1";

export const GRID = 13;
export const MAX_FIXTURES = 6;
export const STEP_BUDGET = 60;
export const MAX_INVALID_ATTEMPTS = 6;
export const MAX_NO_PROGRESS_STREAK = 10;
export const LOOK_COST = 2;

export const COLORS = ["朱红", "靛蓝", "翠绿"] as const;
export const COLOR_HEX = ["#e4573d", "#3f6fd8", "#31a86b"] as const;
export const SHAPES = ["立方", "球体", "锥体"] as const;
export const GLYPHS = ["圆环", "三角", "方框"] as const;
export const GLYPH_SYMBOLS = ["◎", "△", "□"] as const;

export const ACTION_COUNT = MAX_FIXTURES * 3 + 2;
export const LOOK_ACTION = MAX_FIXTURES * 3;
export const REFUSE_ACTION = LOOK_ACTION + 1;

export type ActionKind = "approach" | "inspect" | "operate" | "look" | "refuse";

export interface ActionSpec {
  index: number;
  kind: ActionKind;
  slot: number | null;
  label: string;
  groupLabel: string;
}

export function actionKind(action: number): ActionKind {
  if (action < MAX_FIXTURES) return "approach";
  if (action < MAX_FIXTURES * 2) return "inspect";
  if (action < MAX_FIXTURES * 3) return "operate";
  if (action === LOOK_ACTION) return "look";
  return "refuse";
}

export function actionSlot(action: number): number | null {
  return action < MAX_FIXTURES * 3 ? action % MAX_FIXTURES : null;
}

export const ACTIONS: ActionSpec[] = [
  ...Array.from({ length: MAX_FIXTURES }, (_, i) => ({
    index: i,
    kind: "approach" as const,
    slot: i,
    label: `接近物件区 #${i}`,
    groupLabel: "接近",
  })),
  ...Array.from({ length: MAX_FIXTURES }, (_, i) => ({
    index: MAX_FIXTURES + i,
    kind: "inspect" as const,
    slot: i,
    label: `检查标记 #${i}`,
    groupLabel: "检查",
  })),
  ...Array.from({ length: MAX_FIXTURES }, (_, i) => ({
    index: MAX_FIXTURES * 2 + i,
    kind: "operate" as const,
    slot: i,
    label: `操作装置 #${i}`,
    groupLabel: "操作",
  })),
  { index: LOOK_ACTION, kind: "look", slot: null, label: "请求额外视野（读取模糊标记）", groupLabel: "视野" },
  { index: REFUSE_ACTION, kind: "refuse", slot: null, label: "拒绝执行（安全放弃）", groupLabel: "终止" },
];

export const actionLabel = (a: number): string => ACTIONS[a]?.label ?? `未知动作 ${a}`;

export const TRAIN_SEED_START = 1000;
export const TRAIN_SEED_END = 1279;
export const DEV_SEED_START = 1280;
export const DEV_SEED_END = 1499;
export const VAL_SEED_START = 2000;
export const VAL_SEED_END = 2079;
export const HIDDEN_STANDARD_START = 9000;
export const HIDDEN_STANDARD_END = 9099;
export const HIDDEN_AMBIGUOUS_START = 9300;
export const HIDDEN_AMBIGUOUS_END = 9399;
export const HIDDEN_CLUTTERED_START = 9400;
export const HIDDEN_CLUTTERED_END = 9499;
export const HIDDEN_UNSOLVABLE_START = 9500;
export const HIDDEN_UNSOLVABLE_END = 9524;
export const TRAIN_REFUSAL_START = 1600;
export const TRAIN_REFUSAL_END = 1699;
export const VAL_REFUSAL_START = 2600;
export const VAL_REFUSAL_END = 2619;
/**
 * Held-out "new layout" family: same observable schema and action set, but the
 * generator uses a different spawn corner, denser clutter and tighter fixture
 * spacing. Never used for training, threshold calibration or model selection.
 */
export const NOVEL_LAYOUT_START = 3000;
export const NOVEL_LAYOUT_END = 3099;

export type Condition = "standard" | "ambiguous" | "cluttered" | "unsolvable";

export function hiddenConditionForSeed(seed: number): Condition | null {
  if (seed >= NOVEL_LAYOUT_START && seed <= NOVEL_LAYOUT_END) return "standard";
  if (seed >= HIDDEN_AMBIGUOUS_START && seed <= HIDDEN_AMBIGUOUS_END) return "ambiguous";
  if (seed >= HIDDEN_CLUTTERED_START && seed <= HIDDEN_CLUTTERED_END) return "cluttered";
  if (seed >= HIDDEN_UNSOLVABLE_START && seed <= HIDDEN_UNSOLVABLE_END) return "unsolvable";
  if (seed >= HIDDEN_STANDARD_START && seed <= HIDDEN_STANDARD_END) return "standard";
  return null;
}

export function isNovelLayoutSeed(seed: number): boolean {
  return seed >= NOVEL_LAYOUT_START && seed <= NOVEL_LAYOUT_END;
}

export function seedFamily(seed: number): "train" | "dev" | "validation" | "novel" | "hidden-eval" | "manual" {
  if (seed >= TRAIN_SEED_START && seed <= TRAIN_SEED_END) return "train";
  if (seed >= TRAIN_REFUSAL_START && seed <= TRAIN_REFUSAL_END) return "train";
  if (seed >= DEV_SEED_START && seed <= DEV_SEED_END) return "dev";
  if (seed >= VAL_SEED_START && seed <= VAL_SEED_END) return "validation";
  if (seed >= VAL_REFUSAL_START && seed <= VAL_REFUSAL_END) return "validation";
  if (isNovelLayoutSeed(seed)) return "novel";
  if (hiddenConditionForSeed(seed)) return "hidden-eval";
  return "manual";
}

// ---- observation schema -------------------------------------------------
// Per slot (20): present, device, color(3), shape(3), glyph(3, known only),
// glyphKnown, smudged, distance, reachable, adjacent, colorMatch, shapeMatch,
// glyphMatch, candidate.
// Global: goal color(3), goal shape(3), goal glyph(3), inspected(6),
// looked, self(2), stepsRemaining.

export const PER_SLOT_FEATURES = 20;
const SLOT_BLOCK = MAX_FIXTURES * PER_SLOT_FEATURES;
export const GLOBAL_START = SLOT_BLOCK;
export const GOAL_COLOR_START = GLOBAL_START;
export const GOAL_SHAPE_START = GOAL_COLOR_START + 3;
export const GOAL_GLYPH_START = GOAL_SHAPE_START + 3;
export const INSPECTED_START = GOAL_GLYPH_START + 3;
export const LOOKED_START = INSPECTED_START + MAX_FIXTURES;
export const SELF_START = LOOKED_START + 1;
export const STEPS_START = SELF_START + 2;
export const FEATURE_DIM = STEPS_START + 1;

export const SLOT_FEATURE_FIELDS = [
  { key: "present", label: "存在", start: 0, size: 1 },
  { key: "device", label: "装置", start: 1, size: 1 },
  { key: "color", label: "颜色", start: 2, size: 3 },
  { key: "shape", label: "形状", start: 5, size: 3 },
  { key: "glyph", label: "已知刻符", start: 8, size: 3 },
  { key: "glyphKnown", label: "刻符已知", start: 11, size: 1 },
  { key: "smudged", label: "标记模糊", start: 12, size: 1 },
  { key: "distance", label: "路径距离", start: 13, size: 1 },
  { key: "reachable", label: "可达", start: 14, size: 1 },
  { key: "adjacent", label: "相邻", start: 15, size: 1 },
  { key: "colorMatch", label: "颜色相符", start: 16, size: 1 },
  { key: "shapeMatch", label: "形状相符", start: 17, size: 1 },
  { key: "glyphMatch", label: "刻符相符", start: 18, size: 1 },
  { key: "candidate", label: "候选装置", start: 19, size: 1 },
] as const;

export const FEATURE_FIELDS = [
  ...Array.from({ length: MAX_FIXTURES }, (_, s) =>
    SLOT_FEATURE_FIELDS.map((f) => ({
      key: `slot${s}.${f.key}`,
      label: `槽#${s} ${f.label}`,
      start: s * PER_SLOT_FEATURES + f.start,
      size: f.size,
    })),
  ).flat(),
  { key: "goal.color", label: "目标颜色", start: GOAL_COLOR_START, size: 3 },
  { key: "goal.shape", label: "目标形状", start: GOAL_SHAPE_START, size: 3 },
  { key: "goal.glyph", label: "目标刻符", start: GOAL_GLYPH_START, size: 3 },
  { key: "inspected", label: "已检查槽位", start: INSPECTED_START, size: 6 },
  { key: "looked", label: "已请求视野", start: LOOKED_START, size: 1 },
  { key: "self", label: "角色坐标", start: SELF_START, size: 2 },
  { key: "steps", label: "剩余步数", start: STEPS_START, size: 1 },
];

export const slotFieldStart = (slot: number, key: string): number | null => {
  const f = SLOT_FEATURE_FIELDS.find((x) => x.key === key);
  return f ? slot * PER_SLOT_FEATURES + f.start : null;
};

export const argmax = (values: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
};
