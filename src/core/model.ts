import { ACTION_COUNT, FEATURE_DIM } from "./constants";
import { mulberry32, hashString, type Rng } from "./rng";

export interface ModelMeta {
  id: string;
  version: string;
  featureVersion: string;
  actionVersion: string;
  datasetVersion: string;
  createdAt: string;
  seeds: number[];
  arch: number[];
  trainSamples: number;
  accuracy: number;
  loss: number;
  notes: string;
}

export interface ModelJSON {
  meta: ModelMeta;
  weights: number[][][];
  biases: number[][];
}

export class MLP {
  sizes: number[];
  weights: number[][][] = [];
  biases: number[][] = [];

  constructor(sizes: number[], rng?: Rng) {
    this.sizes = sizes;
    for (let l = 0; l < sizes.length - 1; l++) {
      const inSize = sizes[l];
      const outSize = sizes[l + 1];
      const scale = Math.sqrt(2 / inSize);
      const w: number[][] = [];
      const b: number[] = [];
      for (let o = 0; o < outSize; o++) {
        const row = new Array<number>(inSize);
        for (let i = 0; i < inSize; i++) row[i] = rng ? (rng.next() * 2 - 1) * scale : 0;
        w.push(row);
        b.push(0);
      }
      this.weights.push(w);
      this.biases.push(b);
    }
  }

  static fromJSON(json: ModelJSON): MLP {
    const net = new MLP(json.meta.arch);
    net.weights = json.weights;
    net.biases = json.biases;
    return net;
  }

  toJSON(meta: ModelMeta): ModelJSON {
    return { meta, weights: this.weights, biases: this.biases };
  }

  forward(input: ArrayLike<number>): number[] {
    let act: number[] = Array.from(input);
    for (let l = 0; l < this.weights.length; l++) {
      const w = this.weights[l];
      const b = this.biases[l];
      const out = new Array<number>(w.length).fill(0);
      for (let o = 0; o < w.length; o++) {
        const row = w[o];
        let sum = b[o];
        for (let i = 0; i < row.length; i++) sum += row[i] * act[i];
        out[o] = sum;
      }
      if (l < this.weights.length - 1) {
        for (let i = 0; i < out.length; i++) out[i] = out[i] > 0 ? out[i] : 0;
      }
      act = out;
    }
    return act;
  }
}

export function initModel(seed: number): MLP {
  const rng = mulberry32(seed);
  return new MLP([FEATURE_DIM, 96, 96, ACTION_COUNT], rng);
}

/**
 * Single-layer linear softmax policy over the same observable features. It uses
 * the shared optimizer with one dense layer, so the comparison isolates
 * representation capacity (linear vs. one hidden ReLU stack), not the trainer.
 */
export function initLinearModel(seed: number): MLP {
  const rng = mulberry32(seed);
  return new MLP([FEATURE_DIM, ACTION_COUNT], rng);
}

export function paramCount(net: MLP): number {
  let total = 0;
  for (let l = 0; l < net.weights.length; l++) {
    total += net.weights[l].length * net.biases[l].length;
  }
  return total;
}

/** Multiply-accumulate operations of one forward pass (biases counted as adds). */
export function macCount(net: MLP): number {
  let total = 0;
  for (const layer of net.weights) {
    total += layer.length * layer[0].length;
  }
  return total;
}

export interface TrainSample {
  x: number[];
  y: number;
  weight: number;
}

export interface TrainConfig {
  epochs: number;
  batchSize: number;
  lr: number;
  seed: number;
  l2: number;
}

export interface TrainResult {
  losses: number[];
  accuracy: number;
  valAccuracy: number;
}

function softmax(logits: number[]): number[] {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function trainModel(
  net: MLP,
  samples: TrainSample[],
  config: TrainConfig,
  onEpoch?: (epoch: number, loss: number, acc: number) => void,
): TrainResult {
  const rng = mulberry32(config.seed);
  const order = samples.map((_, i) => i);
  const m = net.weights.map((layer) => layer.map((row) => row.map(() => 0)));
  const v = net.weights.map((layer) => layer.map((row) => row.map(() => 0)));
  const mb = net.biases.map((b) => b.map(() => 0));
  const vb = net.biases.map((b) => b.map(() => 0));
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  let t = 0;
  const losses: number[] = [];

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    rng.shuffle(order);
    let totalLoss = 0;
    let correct = 0;
    let seen = 0;
    for (let start = 0; start < order.length; start += config.batchSize) {
      const batch = order.slice(start, start + config.batchSize);
      const gradW = net.weights.map((layer) => layer.map((row) => row.map(() => 0)));
      const gradB = net.biases.map((b) => b.map(() => 0));
      for (const idx of batch) {
        const { x, y, weight } = samples[idx];
        const acts: number[][] = [Array.from(x)];
        for (let l = 0; l < net.weights.length; l++) {
          const w = net.weights[l];
          const b = net.biases[l];
          const prev = acts[l];
          const out = new Array<number>(w.length).fill(0);
          for (let o = 0; o < w.length; o++) {
            let sum = b[o];
            for (let i = 0; i < prev.length; i++) sum += w[o][i] * prev[i];
            out[o] = sum;
          }
          if (l < net.weights.length - 1) for (let i = 0; i < out.length; i++) out[i] = Math.max(0, out[i]);
          acts.push(out);
        }
        const probs = softmax(acts[acts.length - 1]);
        totalLoss += -Math.log(Math.max(probs[y], 1e-12)) * weight;
        if (argmaxArr(probs) === y) correct += 1;
        seen += weight;
        const delta: number[][] = new Array(net.weights.length);
        const last = new Array(probs.length).fill(0);
        for (let k = 0; k < probs.length; k++) last[k] = (probs[k] - (k === y ? 1 : 0)) * weight;
        delta[delta.length - 1] = last;
        for (let l = net.weights.length - 2; l >= 0; l--) {
          const dNext = delta[l + 1];
          const wNext = net.weights[l + 1];
          const prevAct = acts[l + 1];
          const d = new Array<number>(prevAct.length).fill(0);
          for (let o = 0; o < wNext.length; o++) {
            if (prevAct[o] <= 0) continue;
            const g = dNext[o];
            if (g === 0) continue;
            const row = wNext[o];
            for (let i = 0; i < row.length; i++) d[i] += row[i] * g;
          }
          delta[l] = d;
        }
        for (let l = 0; l < net.weights.length; l++) {
          const d = delta[l];
          const prev = acts[l];
          for (let o = 0; o < net.weights[l].length; o++) {
            if (l < net.weights.length - 1 && acts[l + 1][o] <= 0) continue;
            const g = d[o];
            if (g === 0) continue;
            const gb = gradB[l];
            const gw = gradW[l][o];
            gb[o] += g;
            for (let i = 0; i < prev.length; i++) if (prev[i] !== 0) gw[i] += g * prev[i];
          }
        }
      }
      t++;
      const scale = 1 / batch.length;
      for (let l = 0; l < net.weights.length; l++) {
        for (let o = 0; o < net.weights[l].length; o++) {
          const gw = gradW[l][o];
          for (let i = 0; i < gw.length; i++) {
            let g = gw[i] * scale;
            if (config.l2 > 0) g += config.l2 * net.weights[l][o][i];
            m[l][o][i] = beta1 * m[l][o][i] + (1 - beta1) * g;
            v[l][o][i] = beta2 * v[l][o][i] + (1 - beta2) * g * g;
            const mh = m[l][o][i] / (1 - Math.pow(beta1, t));
            const vh = v[l][o][i] / (1 - Math.pow(beta2, t));
            net.weights[l][o][i] -= (config.lr * mh) / (Math.sqrt(vh) + eps);
          }
          let gb = gradB[l][o] * scale;
          mb[l][o] = beta1 * mb[l][o] + (1 - beta1) * gb;
          vb[l][o] = beta2 * vb[l][o] + (1 - beta2) * gb * gb;
          const mhb = mb[l][o] / (1 - Math.pow(beta1, t));
          const vhb = vb[l][o] / (1 - Math.pow(beta2, t));
          net.biases[l][o] -= (config.lr * mhb) / (Math.sqrt(vhb) + eps);
        }
      }
    }
    const loss = totalLoss / Math.max(1, seen);
    losses.push(loss);
    onEpoch?.(epoch, loss, correct / Math.max(1, seen));
  }

  return { losses, accuracy: evaluateAccuracy(net, samples), valAccuracy: 0 };
}

export function evaluateAccuracy(net: MLP, samples: TrainSample[]): number {
  if (samples.length === 0) return 0;
  let ok = 0;
  const limit = Math.min(samples.length, 2000);
  for (let i = 0; i < limit; i++) {
    const out = net.forward(samples[i].x);
    if (argmaxArr(out) === samples[i].y) ok++;
  }
  return ok / limit;
}

function argmaxArr(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

export function modelVersion(net: MLP, meta: Omit<ModelMeta, "version">): string {
  const payload = JSON.stringify({ w: net.weights, b: net.biases, seeds: meta.seeds });
  return `${meta.id}-${hashString(payload).toString(16).padStart(8, "0")}`;
}
