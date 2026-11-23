import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { COLOR_HEX, GLYPH_SYMBOLS, GRID, MAX_FIXTURES } from "../core/constants";
import type { Scenario, SlotView } from "../core/types";

const TILE = 1;
const WORLD_OFFSET = (GRID - 1) / 2;

interface FixtureVisual {
  root: THREE.Group;
  body: THREE.Mesh;
  ring: THREE.Mesh;
  glyph: THREE.Sprite;
  platform: THREE.Mesh;
  beacons: THREE.Mesh[];
  lastGlyphText: string | null;
  lastKind: string | null;
}

interface CharacterVisual {
  root: THREE.Group;
  body: THREE.Mesh;
  marker: THREE.Mesh;
}

export interface SlotHighlightState {
  slots: SlotView[];
  inspected: number[];
  candidateSlot: number | null;
  adjacentSlot: number | null;
  revealed: boolean;
}

export class Scene3D {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  private fixtureVisuals: FixtureVisual[] = [];
  private character!: CharacterVisual;
  private pathLine: THREE.Line;
  private pulseRings: THREE.Mesh[] = [];
  private sweep: THREE.Mesh;
  private sweepT = 1;
  animationSpeed = 1.6;
  private animation: { path: Array<{ x: number; y: number }>; index: number; from: THREE.Vector3; to: THREE.Vector3; t: number } | null =
    null;
  private pendingResolvers: Array<() => void> = [];
  private glyphTextures = new Map<string, THREE.Texture>();
  private clock = new THREE.Clock();
  private disposed = false;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#0b1020");
    this.scene.fog = new THREE.Fog("#0b1020", 20, 36);

    this.camera = new THREE.PerspectiveCamera(46, 1, 0.1, 100);
    this.camera.position.set(0, 13.5, 13.5);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.6, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.46;
    this.controls.minDistance = 7;
    this.controls.maxDistance = 28;

    this.buildEnvironment();

    const pathGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    this.pathLine = new THREE.Line(
      pathGeo,
      new THREE.LineBasicMaterial({ color: "#ffd166", transparent: true, opacity: 0.85 }),
    );
    this.pathLine.visible = false;
    this.scene.add(this.pathLine);

    this.sweep = new THREE.Mesh(
      new THREE.RingGeometry(0.6, 0.75, 64),
      new THREE.MeshBasicMaterial({ color: "#7dd3fc", transparent: true, opacity: 0, side: THREE.DoubleSide }),
    );
    this.sweep.rotation.x = -Math.PI / 2;
    this.sweep.position.y = 0.08;
    this.scene.add(this.sweep);

    this.resize();
    window.addEventListener("resize", this.resize);
    this.renderer.setAnimationLoop(() => this.tick());
  }

  private buildEnvironment(): void {
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(GRID, GRID),
      new THREE.MeshStandardMaterial({ color: "#131a2e", roughness: 0.9, metalness: 0.05 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const tileGeo = new THREE.PlaneGeometry(TILE * 0.94, TILE * 0.94);
    for (let y = 0; y < GRID; y++) {
      for (let x = 0; x < GRID; x++) {
        const mat = new THREE.MeshStandardMaterial({
          color: (x + y) % 2 === 0 ? "#18223a" : "#151e33",
          roughness: 0.85,
          transparent: true,
          opacity: 0.9,
        });
        const tile = new THREE.Mesh(tileGeo, mat);
        tile.rotation.x = -Math.PI / 2;
        tile.position.set(x - WORLD_OFFSET, 0.01, y - WORLD_OFFSET);
        this.scene.add(tile);
      }
    }

    const grid = new THREE.GridHelper(GRID, GRID, "#2a3a5c", "#1d2942");
    grid.position.y = 0.02;
    this.scene.add(grid);

    const hemi = new THREE.HemisphereLight("#9fb6ff", "#1a1f2e", 0.85);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight("#ffffff", 2.1);
    key.position.set(6, 14, 7);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -10;
    key.shadow.camera.right = 10;
    key.shadow.camera.top = 10;
    key.shadow.camera.bottom = -10;
    this.scene.add(key);

    const rim = new THREE.DirectionalLight("#5b8cff", 0.9);
    rim.position.set(-8, 8, -6);
    this.scene.add(rim);

    const glow = new THREE.PointLight("#ffb703", 8, 12, 2);
    glow.position.set(0, 4, 0);
    this.scene.add(glow);

    const character = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.28, 0.45, 8, 16),
      new THREE.MeshStandardMaterial({ color: "#ffd166", roughness: 0.4, metalness: 0.2, emissive: "#5a3d00", emissiveIntensity: 0.5 }),
    );
    body.position.y = 0.62;
    body.castShadow = true;
    character.add(body);
    const visor = new THREE.Mesh(
      new THREE.BoxGeometry(0.34, 0.12, 0.1),
      new THREE.MeshStandardMaterial({ color: "#0b1020", emissive: "#7dd3fc", emissiveIntensity: 0.8 }),
    );
    visor.position.set(0, 0.8, 0.24);
    character.add(visor);
    const marker = new THREE.Mesh(
      new THREE.ConeGeometry(0.12, 0.3, 16),
      new THREE.MeshStandardMaterial({ color: "#7dd3fc", emissive: "#38bdf8", emissiveIntensity: 0.9 }),
    );
    marker.rotation.x = Math.PI;
    marker.position.y = 1.25;
    character.add(marker);
    this.scene.add(character);
    this.character = { root: character, body, marker };
  }

  resize = (): void => {
    if (this.disposed) return;
    const canvas = this.renderer.domElement;
    const parent = canvas.parentElement;
    const width = parent?.clientWidth ?? window.innerWidth;
    const height = parent?.clientHeight ?? window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
  };

  private glyphTexture(symbol: string, color: string): THREE.Texture {
    const key = `${symbol}:${color}`;
    const cached = this.glyphTextures.get(key);
    if (cached) return cached;
    const size = 128;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "rgba(10,16,32,0.85)";
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size * 0.42, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "bold 62px 'Segoe UI Symbol', system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(symbol, size / 2, size / 2 + 4);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    this.glyphTextures.set(key, texture);
    return texture;
  }

  setScenario(scenario: Scenario): void {
    for (const f of this.fixtureVisuals) this.scene.remove(f.root);
    this.fixtureVisuals = [];
    for (const ring of this.pulseRings) this.scene.remove(ring);
    this.pulseRings = [];

    for (const [x, y, wall] of scenario.walls.map((w, i) => [i % GRID, Math.floor(i / GRID), w] as const)) {
      if (!wall) continue;
      const block = new THREE.Mesh(
        new THREE.BoxGeometry(TILE * 0.96, 0.9, TILE * 0.96),
        new THREE.MeshStandardMaterial({ color: "#2c3653", roughness: 0.75, metalness: 0.15 }),
      );
      block.position.set(x - WORLD_OFFSET, 0.45, y - WORLD_OFFSET);
      block.castShadow = true;
      block.receiveShadow = true;
      this.scene.add(block);
    }

    for (const fixture of scenario.fixtures) {
      const root = new THREE.Group();
      root.position.set(fixture.pos.x - WORLD_OFFSET, 0, fixture.pos.y - WORLD_OFFSET);
      const color = COLOR_HEX[fixture.color];

      const platform = new THREE.Mesh(
        new THREE.CylinderGeometry(0.46, 0.52, 0.12, 24),
        new THREE.MeshStandardMaterial({
          color: fixture.kind === "device" ? "#1e293b" : "#334155",
          roughness: 0.5,
          metalness: 0.5,
        }),
      );
      platform.position.y = 0.06;
      platform.receiveShadow = true;
      platform.castShadow = true;
      root.add(platform);

      let bodyGeo: THREE.BufferGeometry;
      if (fixture.shape === 0) bodyGeo = new THREE.BoxGeometry(0.62, 0.62, 0.62);
      else if (fixture.shape === 1) bodyGeo = new THREE.SphereGeometry(0.36, 24, 18);
      else bodyGeo = new THREE.ConeGeometry(0.4, 0.7, 20);
      const body = new THREE.Mesh(
        bodyGeo,
        new THREE.MeshStandardMaterial({
          color,
          roughness: 0.35,
          metalness: fixture.kind === "device" ? 0.55 : 0.2,
          emissive: new THREE.Color(color).multiplyScalar(fixture.kind === "device" ? 0.16 : 0.05),
        }),
      );
      body.position.y = 0.55;
      body.castShadow = true;
      root.add(body);

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.52, 0.035, 10, 48),
        new THREE.MeshStandardMaterial({
          color: fixture.kind === "device" ? "#7dd3fc" : "#475569",
          emissive: new THREE.Color(fixture.kind === "device" ? "#38bdf8" : "#000000"),
          emissiveIntensity: fixture.kind === "device" ? 0.9 : 0,
        }),
      );
      ring.rotation.x = Math.PI / 2;
      ring.position.y = 0.18;
      root.add(ring);

      const glyph = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: this.glyphTexture("?", "#94a3b8"),
          transparent: true,
          depthWrite: false,
        }),
      );
      glyph.scale.set(0.66, 0.66, 0.66);
      glyph.position.y = 1.42;
      root.add(glyph);

      const beacons: THREE.Mesh[] = [];
      for (let b = 0; b < 3; b++) {
        const beacon = new THREE.Mesh(
          new THREE.SphereGeometry(0.05, 8, 8),
          new THREE.MeshBasicMaterial({ color: "#64748b", transparent: true, opacity: 0.6 }),
        );
        beacon.position.set(-0.14 + b * 0.14, 0.2, 0.34);
        root.add(beacon);
        beacons.push(beacon);
      }

      root.userData.slot = fixture.slot;
      this.scene.add(root);
      this.fixtureVisuals.push({
        root,
        body,
        ring,
        glyph,
        platform,
        beacons,
        lastGlyphText: null,
        lastKind: null,
      });
    }
  }

  setCharacterCell(pos: { x: number; y: number }): void {
    this.character.root.position.set(pos.x - WORLD_OFFSET, 0, pos.y - WORLD_OFFSET);
  }

  setPoseBeforeStep(pos: { x: number; y: number }): void {
    this.setCharacterCell(pos);
    this.pathLine.visible = false;
    this.animation = null;
  }

  /** Animates the character along a path; resolves when finished. */
  animateAlong(path: Array<{ x: number; y: number }>, speed = 1.6): Promise<void> {
    if (path.length === 0) return Promise.resolve();
    this.animationSpeed = speed;
    this.setCharacterCell(path[0]);
    this.pathLine.visible = false;
    const points = path.map((p) => new THREE.Vector3(p.x - WORLD_OFFSET, 0.12, p.y - WORLD_OFFSET));
    this.pathLine.geometry.dispose();
    this.pathLine.geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3().copy(this.character.root.position).setY(0.12),
      ...points,
    ]);
    this.pathLine.visible = true;
    return new Promise((resolve) => {
      this.animation = {
        path,
        index: 0,
        from: this.character.root.position.clone(),
        to: points[0],
        t: 0,
      };
      this.pendingResolvers.push(resolve);
    });
  }

  highlight(state: SlotHighlightState): void {
    this.fixtureVisuals.forEach((visual, slot) => {
      const view = state.slots.find((s) => s.slot === slot);
      const isCandidate = state.candidateSlot === slot;
      const isAdjacent = state.adjacentSlot === slot;
      const inspected = state.inspected.includes(slot);

      const bodyMat = visual.body.material as THREE.MeshStandardMaterial;
      if (view && view.present) {
        const base = new THREE.Color(COLOR_HEX[view.color]);
        bodyMat.color.copy(base);
        bodyMat.emissive.copy(base).multiplyScalar(isCandidate ? 0.55 : 0.08);
      }
      const ringMat = visual.ring.material as THREE.MeshStandardMaterial;
      const ringColor = isCandidate ? "#fbbf24" : inspected ? "#34d399" : view?.adjacent ? "#7dd3fc" : "#334155";
      ringMat.color.set(ringColor);
      ringMat.emissive.set(isCandidate ? "#f59e0b" : inspected ? "#10b981" : "#000000");
      ringMat.emissiveIntensity = isCandidate || inspected ? 0.9 : 0;

      if (view && view.present) {
        const symbol = view.glyphKnown && view.glyph !== null ? GLYPH_SYMBOLS[view.glyph] : "?";
        if (visual.lastGlyphText !== symbol) {
          const color = view.glyphKnown && view.glyph !== null ? COLOR_HEX[view.color] : "#94a3b8";
          const mat = visual.glyph.material as THREE.SpriteMaterial;
          mat.map?.dispose();
          mat.map = this.glyphTexture(symbol, color);
          mat.needsUpdate = true;
          visual.lastGlyphText = symbol;
        }
        visual.beacons.forEach((beacon, i) => {
          const mat = beacon.material as THREE.MeshBasicMaterial;
          const on = inspected ? i < 3 : view.glyphKnown ? i < 2 : i < 1;
          mat.color.set(on ? "#34d399" : "#334155");
          mat.opacity = on ? 1 : 0.4;
        });
      }
      void isAdjacent;
    });
  }

  playLookSweep(): void {
    this.sweep.position.x = this.character.root.position.x;
    this.sweep.position.z = this.character.root.position.z;
    this.sweepT = 0;
  }

  private tick(): void {
    if (this.disposed) return;
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const time = this.clock.elapsedTime;

    if (this.animation) {
      const anim = this.animation;
      anim.t += dt * 2.6 * this.animationSpeed;
      const target = anim.to;
      const start = anim.from;
      const t = Math.min(1, anim.t);
      this.character.root.position.lerpVectors(start, target, t);
      this.character.root.position.y = Math.sin(t * Math.PI) * 0.08;
      if (t >= 1) {
        anim.index++;
        if (anim.index >= anim.path.length) {
          this.animation = null;
          this.pathLine.visible = false;
          const resolvers = this.pendingResolvers.splice(0);
          for (const resolve of resolvers) resolve();
        } else {
          anim.from = this.character.root.position.clone();
          anim.to = new THREE.Vector3(
            anim.path[anim.index].x - WORLD_OFFSET,
            0.1,
            anim.path[anim.index].y - WORLD_OFFSET,
          );
          anim.t = 0;
        }
      }
    }

    this.sweepT = Math.min(1, this.sweepT + dt * 1.6);
    const sweepMat = this.sweep.material as THREE.MeshBasicMaterial;
    if (this.sweepT < 1) {
      this.sweep.visible = true;
      const scale = 1 + this.sweepT * 9;
      this.sweep.scale.set(scale, scale, 1);
      sweepMat.opacity = (1 - this.sweepT) * 0.8;
    } else {
      this.sweep.visible = false;
    }

    this.fixtureVisuals.forEach((visual, slot) => {
      visual.ring.rotation.z = time * 0.6 + slot;
      if (visual.root.userData.slot === undefined) return;
      const bob = Math.sin(time * 1.6 + slot * 1.3) * 0.04;
      visual.glyph.position.y = 1.42 + bob;
    });
    this.character.marker.rotation.y = time * 2.2;
    this.character.body.position.y = 0.62 + Math.sin(time * 2.4) * 0.03;

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    window.removeEventListener("resize", this.resize);
    this.controls.dispose();
    this.renderer.dispose();
  }
}

export const slotWorld = (slot: number, scenario: Scenario): THREE.Vector3 => {
  const f = scenario.fixtures[slot];
  return new THREE.Vector3(f.pos.x - WORLD_OFFSET, 0.6, f.pos.y - WORLD_OFFSET);
};

export const fixtureCount = MAX_FIXTURES;
