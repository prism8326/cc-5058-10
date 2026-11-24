import { FEATURE_DIM, ACTION_COUNT, FEATURE_VERSION, ACTION_VERSION } from "./constants.js";
import type { ModelJSON } from "./model.js";

export const DEFAULT_ARCH = [FEATURE_DIM, 96, 96, ACTION_COUNT];

export interface FormatCheck {
  ok: boolean;
  errors: string[];
}

/** Strict shape/version validation for weights loaded from disk. */
export function validateModelFormat(json: unknown, expectedArch: number[] = DEFAULT_ARCH): FormatCheck {
  const errors: string[] = [];
  const m = json as Partial<ModelJSON>;
  if (!m || typeof m !== "object") return { ok: false, errors: ["模型不是对象"] };
  if (!m.meta) errors.push("缺少 meta");
  if (!Array.isArray(m.weights)) errors.push("缺少 weights");
  if (!Array.isArray(m.biases)) errors.push("缺少 biases");
  if (errors.length) return { ok: false, errors };

  const meta = m.meta!;
  if (meta.featureVersion !== FEATURE_VERSION) errors.push(`特征版本不匹配: ${meta.featureVersion}`);
  if (meta.actionVersion !== ACTION_VERSION) errors.push(`动作版本不匹配: ${meta.actionVersion}`);
  const weights = m.weights!;
  const biases = m.biases!;
  if (weights.length !== biases.length) errors.push("权重/偏置层数不一致");
  if (!Array.isArray(meta.arch) || meta.arch.length !== expectedArch.length) {
    errors.push(`网络结构应为 ${expectedArch.join("→")}`);
  } else {
    for (let i = 0; i < expectedArch.length; i++) {
      if (meta.arch[i] !== expectedArch[i]) {
        errors.push(`网络结构应为 ${expectedArch.join("→")}，实际 ${meta.arch.join("→")}`);
        break;
      }
    }
  }
  for (let l = 0; l < weights.length; l++) {
    const layer = weights[l];
    if (!Array.isArray(layer)) {
      errors.push(`第 ${l} 层权重不是数组`);
      continue;
    }
    if (layer.length !== biases[l]?.length) errors.push(`第 ${l} 层输出维度不一致`);
    for (let o = 0; o < layer.length; o++) {
      const row = layer[o];
      if (!Array.isArray(row)) {
        errors.push(`第 ${l} 层第 ${o} 行不是数组`);
        continue;
      }
      const expectIn = l === 0 ? FEATURE_DIM : weights[l - 1].length;
      if (row.length !== expectIn) errors.push(`第 ${l} 层第 ${o} 行输入维度 ${row.length} != ${expectIn}`);
      if (row.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
        errors.push(`第 ${l} 层第 ${o} 行含非法数值`);
      }
    }
  }
  return { ok: errors.length === 0, errors };
}
