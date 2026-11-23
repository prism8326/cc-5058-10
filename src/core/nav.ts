import type { GridPos, Scenario } from "./types";

export interface NavGrid {
  gridSize: number;
  walls: boolean[];
  blocked: Set<number>;
}

export const cellIndex = (x: number, y: number, n: number): number => y * n + x;

export function passable(grid: NavGrid, x: number, y: number): boolean {
  const n = grid.gridSize;
  if (x < 1 || y < 1 || x >= n - 1 || y >= n - 1) return false;
  return !grid.walls[cellIndex(x, y, n)];
}

const DIRS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
];

export function openNeighbors(grid: NavGrid, pos: GridPos): GridPos[] {
  const out: GridPos[] = [];
  for (const [dx, dy] of DIRS) {
    const x = pos.x + dx;
    const y = pos.y + dy;
    if (passable(grid, x, y)) out.push({ x, y });
  }
  return out;
}

export function shortestPath(grid: NavGrid, start: GridPos, goal: GridPos): GridPos[] | null {
  const n = grid.gridSize;
  if (start.x === goal.x && start.y === goal.y) return [];
  if (!passable(grid, start.x, start.y) || !passable(grid, goal.x, goal.y)) return null;
  const si = cellIndex(start.x, start.y, n);
  const gi = cellIndex(goal.x, goal.y, n);
  const prev = new Int32Array(n * n).fill(-1);
  prev[si] = si;
  const queue: number[] = [si];
  for (let head = 0; head < queue.length; head++) {
    const cur = queue[head];
    const cx = cur % n;
    const cy = (cur - cx) / n;
    for (const [dx, dy] of DIRS) {
      const x = cx + dx;
      const y = cy + dy;
      if (!passable(grid, x, y)) continue;
      const ni = cellIndex(x, y, n);
      if (prev[ni] !== -1) continue;
      prev[ni] = cur;
      if (ni === gi) {
        const path: GridPos[] = [];
        let node = ni;
        while (node !== si) {
          const nx = node % n;
          const ny = (node - nx) / n;
          path.push({ x: nx, y: ny });
          node = prev[node];
        }
        path.reverse();
        return path;
      }
      queue.push(ni);
    }
  }
  return null;
}

export interface Approach {
  distance: number;
  path: GridPos[];
  stand: GridPos;
}

/** Cheapest path from `pos` to any open neighbour of `target`. */
export function nearestApproach(grid: NavGrid, pos: GridPos, target: GridPos): Approach | null {
  let best: Approach | null = null;
  for (const nb of openNeighbors(grid, target)) {
    const path = shortestPath(grid, pos, nb);
    if (!path) continue;
    if (!best || path.length < best.distance) {
      best = { distance: path.length, path, stand: nb };
    }
  }
  return best;
}

export function isAdjacent(pos: GridPos, target: GridPos): boolean {
  return Math.abs(pos.x - target.x) + Math.abs(pos.y - target.y) === 1;
}

export function makeNav(scenario: Pick<Scenario, "gridSize" | "walls">): NavGrid {
  return { gridSize: scenario.gridSize, walls: scenario.walls, blocked: new Set() };
}
