import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import {
  DEFAULT_DIRECTION_MIX,
  MEASUREMENT_CATALOG,
  PLANNER_RULES,
  PLANNER_SCHEMA_VERSION,
  SEMANTIC_LANDMARK_COUNT,
  SEMANTIC_LANDMARKS,
  analyzeFace,
  createInteractiveMorphPlan,
  createMorphPlan,
  semanticOverlayIndices,
  type DirectionMix,
  type FaceAnalysis,
  type FaceObservation,
  type MorphPlan,
  type PlannerEvidenceFamily,
  type Point,
} from "./face-intelligence";

export {
  DEFAULT_DIRECTION_MIX,
  MEASUREMENT_CATALOG,
  PLANNER_RULES,
  PLANNER_SCHEMA_VERSION,
  SEMANTIC_LANDMARK_COUNT,
  SEMANTIC_LANDMARKS,
  analyzeFace,
  createInteractiveMorphPlan,
  createMorphPlan,
};
export type { DirectionMix, FaceAnalysis, FaceObservation, MorphPlan, PlannerEvidenceFamily, Point };
type Triangle = [number, number, number];

export type MorphResult = {
  canvas: HTMLCanvasElement;
  movedRegions: string[];
  maxMovementPx: number;
  safetyScale: number;
  perRegionScale: Record<string, number>;
  safetyStatus: "passed" | "weakened" | "identity";
  plan: MorphPlan;
};

const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
  379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
  127, 162, 21, 54, 103, 67, 109,
];

let landmarkerPromise: Promise<FaceLandmarker> | null = null;
let canonicalFaceTriangles: Triangle[] | null = null;

const publicAsset = (path: string) => {
  const base =
    (import.meta as ImportMeta & { env?: { BASE_URL?: string } }).env?.BASE_URL ?? "/";
  return `${base.endsWith("/") ? base : `${base}/`}${path.replace(/^\//, "")}`;
};

export async function createFaceLandmarker(): Promise<FaceLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import(
        "@mediapipe/tasks-vision"
      );
      if (!canonicalFaceTriangles) {
        initializeFaceTopology(FaceLandmarker.FACE_LANDMARKS_TESSELATION);
      }
      const vision = await FilesetResolver.forVisionTasks(
        publicAsset("mediapipe/wasm")
      );
      const common = {
        baseOptions: {
          modelAssetPath: publicAsset("models/face_landmarker.task"),
        },
        runningMode: "IMAGE" as const,
        numFaces: 1,
        minFaceDetectionConfidence: 0.65,
        minFacePresenceConfidence: 0.65,
        minTrackingConfidence: 0.65,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      };
      try {
        return await FaceLandmarker.createFromOptions(vision, {
          ...common,
          baseOptions: { ...common.baseOptions, delegate: "GPU" },
        });
      } catch {
        return FaceLandmarker.createFromOptions(vision, common);
      }
    })().catch((error) => {
      landmarkerPromise = null;
      throw error;
    });
  }
  return landmarkerPromise;
}

export async function detectFace(image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): Promise<FaceObservation> {
  const landmarker = await createFaceLandmarker();
  const result = landmarker.detect(image);
  const face = result.faceLandmarks?.[0];
  if (!face || face.length < 468) {
    throw new Error("We couldn’t find one clear face in this photo.");
  }
  const categories = result.faceBlendshapes?.[0]?.categories ?? [];
  const blendshapes = Object.fromEntries(categories.map((category) => [category.categoryName, category.score]));
  const matrix = result.facialTransformationMatrixes?.[0];
  return {
    landmarks: face.map((point) => ({ x: point.x, y: point.y, z: point.z })),
    blendshapes,
    transformationMatrix: matrix ? { rows: matrix.rows, columns: matrix.columns, data: Array.from(matrix.data) } : undefined,
  };
}

export async function detectLandmarks(image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): Promise<Point[]> {
  return (await detectFace(image)).landmarks;
}

type Control = { source: Point; dx: number; dy: number; radius: number; region: string };

const pair = (left: number, right: number): [number, number] => [left, right];
const SYMMETRY_PAIRS = [
  pair(33, 263),
  pair(133, 362),
  pair(70, 300),
  pair(105, 334),
  pair(61, 291),
  pair(78, 308),
  pair(234, 454),
  pair(172, 397),
  pair(136, 365),
];

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

function createPoseFrame(points: Point[], width: number, height: number) {
  const point = (index: number) => ({
    x: points[index].x * width,
    y: points[index].y * height,
  });
  const leftEye = point(33);
  const rightEye = point(263);
  const eyeSpan = Math.max(Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y), 0.001);
  const ux = (rightEye.x - leftEye.x) / eyeSpan;
  const uy = (rightEye.y - leftEye.y) / eyeSpan;
  const vx = -uy;
  const vy = ux;
  const forehead = point(10);
  const chin = point(152);
  const origin = {
    x: (forehead.x + chin.x) / 2,
    y: (forehead.y + chin.y) / 2,
  };
  const localPoint = (current: Point) => {
    const x = current.x - origin.x;
    const y = current.y - origin.y;
    return { u: x * ux + y * uy, v: x * vx + y * vy };
  };
  const local = (index: number) => localPoint(point(index));
  const delta = (du: number, dv: number) => ({
    dx: du * ux + dv * vx,
    dy: du * uy + dv * vy,
  });
  return { point, local, localPoint, delta };
}

function buildControls(
  points: Point[],
  width: number,
  height: number,
  plan: MorphPlan,
  strength: number,
): Control[] {
  const controls: Control[] = [];
  const frame = createPoseFrame(points, width, height);
  const p = frame.point;
  const faceWidth = Math.max(
    Math.abs(frame.local(454).u - frame.local(234).u),
    1,
  );
  const faceHeight = Math.max(
    Math.abs(frame.local(152).v - frame.local(10).v),
    1,
  );
  const isProfilePlan = (
    plan.evidenceFamilies?.some((family) => family.id.includes(".profile-")) ?? false
  ) || faceWidth / Math.max(faceHeight * 0.65, 1) < 0.62;
  // Candidate labels describe planner certainty; they are not another hidden
  // strength control. Geometry backoff below is the only post-plan attenuator.
  // Compile the full bounded plan at a useful photographic scale. Regional
  // caps and geometry validation remain the hard limits; this coefficient only
  // prevents well-supported actions from dying as sub-pixel motion.
  // Cheek width collapses naturally as a face turns. Retain a fraction of
  // face height so a supported profile does not silently receive half-sized
  // caps and falloff radii.
  const faceScale = Math.max(faceWidth, faceHeight * 0.65);
  // Full-profile topology contains predicted hidden-side vertices that nearly
  // overlap in 2D. Keep that narrower silhouette on the proven projected-width
  // path; frontal and three-quarter plans use the pose-stable scale.
  const controlScale = isProfilePlan ? faceWidth : faceScale;
  const scale = controlScale * (isProfilePlan ? 0.138 : 0.18) * strength;
  const fallbackCaps: Record<string, number> = {
    Jaw: 0.045,
    Chin: 0.035,
    Nose: 0.022,
    Lips: 0.022,
    Brows: 0.016,
    Symmetry: 0.015,
  };
  const actionFor = (primitive: string) => plan.actions.find((action) => action.primitive === primitive);
  const actionAmount = (primitive: string) => clamp(actionFor(primitive)?.amount ?? 0, -0.9, 0.9);
  const movementLimitFor = (region: string) => {
    const action = plan.actions.find((candidate) => candidate.region === region);
    const plannedCap = action && "maxDisplacement" in action
      ? Number(action.maxDisplacement)
      : Number.NaN;
    const cap = Number.isFinite(plannedCap) ? plannedCap : (fallbackCaps[region] ?? 0.01);
    if (cap <= 0) return 0;
    return controlScale * clamp(cap, 0.004, fallbackCaps[region] ?? 0.024);
  };
  const jawFactor = actionAmount("jaw-width");
  const chinFactor = actionAmount("chin-length");
  const noseFactor = actionAmount("nose-width");
  const lipFactor = actionAmount("mouth-width");
  const browFactor = actionAmount("brow-height");
  const symmetryFactor = actionAmount("paired-alignment");
  const add = (
    index: number,
    dx: number,
    dy: number,
    radius: number,
    region: string,
  ) => {
    const movementLimit = movementLimitFor(region);
    const magnitude = Math.hypot(dx, dy);
    const movementScale = magnitude > movementLimit
      ? movementLimit / Math.max(magnitude, 0.001)
      : 1;
    controls.push({
      source: p(index),
      dx: dx * movementScale,
      dy: dy * movementScale,
      radius,
      region,
    });
  };
  const addLocal = (
    index: number,
    du: number,
    dv: number,
    radius: number,
    region: string,
  ) => {
    const movement = frame.delta(du, dv);
    add(index, movement.dx, movement.dy, radius, region);
  };

  // Compile named primitives into dense-mesh handles. Rules never address raw
  // points directly; they blend first in primitive space, then compile once.
  const jaw = jawFactor * scale;
  const jawWeights = [1, 0.88, 0.66, 0.42, 0.2, 0.06];
  const visualLeftJaw = [172, 136, 150, 149, 176, 148];
  const visualRightJaw = [397, 365, 379, 378, 400, 377];
  const noseU = frame.local(1).u;
  const leftSpan = Math.abs(noseU - frame.local(234).u);
  const rightSpan = Math.abs(frame.local(454).u - noseU);
  const yawSignal = Math.abs(leftSpan - rightSpan) / Math.max(leftSpan + rightSpan, 1);
  const visibilityT = isProfilePlan ? 1 : clamp((yawSignal - 0.25) / 0.4, 0, 1);
  const hiddenSideFade = visibilityT * visibilityT * (3 - 2 * visibilityT);
  const visualLeftWeight = leftSpan >= rightSpan ? 1 : 1 - hiddenSideFade;
  const visualRightWeight = rightSpan >= leftSpan ? 1 : 1 - hiddenSideFade;
  const jawRadius = controlScale * (isProfilePlan ? 0.075 : 0.14);
  if (visualLeftWeight > 0.01) visualLeftJaw.forEach((index, order) =>
    addLocal(index, -jaw * jawWeights[order] * visualLeftWeight, 0, jawRadius, "Jaw"),
  );
  if (visualRightWeight > 0.01) visualRightJaw.forEach((index, order) =>
    addLocal(index, jaw * jawWeights[order] * visualRightWeight, 0, jawRadius, "Jaw"),
  );

  const chin = chinFactor * scale;
  const chinRadius = controlScale * (isProfilePlan ? 0.07 : 0.12);
  addLocal(152, 0, chin * 0.7, chinRadius, "Chin");
  addLocal(148, 0, chin * 0.28, chinRadius * 0.92, "Chin");
  addLocal(377, 0, chin * 0.28, chinRadius * 0.92, "Chin");

  const nose = noseFactor * scale;
  // Pose support is decided by the planner. If a front/three-quarter Nose
  // action exists, compile it even when a separate 2D span heuristic looks
  // profile-like. A wider field avoids concentrating strain in alar slivers.
  if (visualLeftWeight > 0.01) {
    [98, 97, 64, 49].forEach((index, order) => addLocal(index, nose * (0.55 - order * 0.045) * visualLeftWeight, 0, faceScale * 0.09, "Nose"));
  }
  if (visualRightWeight > 0.01) {
    [327, 326, 294, 279].forEach((index, order) => addLocal(index, -nose * (0.55 - order * 0.045) * visualRightWeight, 0, faceScale * 0.09, "Nose"));
  }

  const lipDelta = lipFactor * scale;
  if (Math.abs(lipDelta) > 0.001) {
    [61, 78, 185, 146].forEach((index) => addLocal(index, -lipDelta * 0.35, 0, faceScale * 0.075, "Lips"));
    [291, 308, 409, 375].forEach((index) => addLocal(index, lipDelta * 0.35, 0, faceScale * 0.075, "Lips"));
  }

  const brow = browFactor * scale;
  const browWeights = [0.7, 0.9, 1, 0.78, 0.5];
  if (visualLeftWeight > 0.01) [70, 63, 105, 66, 107].forEach((index, order) =>
    addLocal(index, 0, -brow * 0.45 * browWeights[order] * visualLeftWeight, faceScale * 0.075, "Brows"),
  );
  if (visualRightWeight > 0.01) [300, 293, 334, 296, 336].forEach((index, order) =>
    addLocal(index, 0, -brow * 0.45 * browWeights[order] * visualRightWeight, faceScale * 0.075, "Brows"),
  );

  const poseFade = clamp((0.3 - yawSignal) / 0.22, 0, 1);
  const symmetry = symmetryFactor * strength * 1.6 * poseFade;
  const centerU = (frame.local(10).u + frame.local(152).u + frame.local(1).u) / 3;
  const deadband = faceHeight * 0.0035;
  for (const [li, ri] of SYMMETRY_PAIRS.slice(0, 4)) {
    const left = frame.local(li);
    const right = frame.local(ri);
    const difference = right.v - left.v;
    if (Math.abs(difference) <= deadband) continue;
    const correction = (Math.abs(difference) - deadband) * Math.sign(difference) * symmetry * 0.5;
    addLocal(li, 0, correction, faceScale * 0.08, "Symmetry");
    addLocal(ri, 0, -correction, faceScale * 0.08, "Symmetry");
  }

  const applyCenteredGroup = (indices: number[], groupCenterU: number, amount: number, radius: number) => {
    const rawShift = centerU - groupCenterU;
    if (Math.abs(rawShift) <= deadband) return;
    const shift = (Math.abs(rawShift) - deadband) * Math.sign(rawShift) * amount;
    indices.forEach((index) => addLocal(index, shift, 0, radius, "Symmetry"));
  };
  const lipLandmarks = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415];
  applyCenteredGroup(
    lipLandmarks,
    (frame.local(61).u + frame.local(291).u) / 2,
    symmetry * 0.6,
    faceScale * 0.06,
  );
  applyCenteredGroup(
    [1, 2, 49, 279, 98, 327],
    frame.local(1).u,
    symmetry * 0.42,
    faceScale * 0.065,
  );
  applyCenteredGroup(
    [152, 148, 377],
    frame.local(152).u,
    symmetry * 0.36,
    faceScale * 0.08,
  );

  return controls.filter((control) => Math.hypot(control.dx, control.dy) > 0.01);
}

function drawTriangle(
  ctx: CanvasRenderingContext2D,
  image: CanvasImageSource,
  source: Point[],
  target: Point[],
) {
  const [s0, s1, s2] = source;
  const [d0, d1, d2] = target;
  const determinant =
    s0.x * (s1.y - s2.y) + s1.x * (s2.y - s0.y) + s2.x * (s0.y - s1.y);
  if (Math.abs(determinant) < 0.001) return;

  const a =
    (d0.x * (s1.y - s2.y) + d1.x * (s2.y - s0.y) + d2.x * (s0.y - s1.y)) /
    determinant;
  const c =
    (d0.x * (s2.x - s1.x) + d1.x * (s0.x - s2.x) + d2.x * (s1.x - s0.x)) /
    determinant;
  const e =
    (d0.x * (s1.x * s2.y - s2.x * s1.y) +
      d1.x * (s2.x * s0.y - s0.x * s2.y) +
      d2.x * (s0.x * s1.y - s1.x * s0.y)) /
    determinant;
  const b =
    (d0.y * (s1.y - s2.y) + d1.y * (s2.y - s0.y) + d2.y * (s0.y - s1.y)) /
    determinant;
  const d =
    (d0.y * (s2.x - s1.x) + d1.y * (s0.x - s2.x) + d2.y * (s1.x - s0.x)) /
    determinant;
  const f =
    (d0.y * (s1.x * s2.y - s2.x * s1.y) +
      d1.y * (s2.x * s0.y - s0.x * s2.y) +
      d2.y * (s0.x * s1.y - s1.x * s0.y)) /
    determinant;

  const targetCenter = {
    x: (d0.x + d1.x + d2.x) / 3,
    y: (d0.y + d1.y + d2.y) / 3,
  };
  const clipPoints = target.map((point) => {
    const x = point.x - targetCenter.x;
    const y = point.y - targetCenter.y;
    const length = Math.max(Math.hypot(x, y), 0.001);
    const overdraw = 0.18;
    return {
      x: point.x + (x / length) * overdraw,
      y: point.y + (y / length) * overdraw,
    };
  });

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(clipPoints[0].x, clipPoints[0].y);
  ctx.lineTo(clipPoints[1].x, clipPoints[1].y);
  ctx.lineTo(clipPoints[2].x, clipPoints[2].y);
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(a, b, c, d, e, f);
  ctx.drawImage(image, 0, 0);
  ctx.restore();
}

const signedArea = (a: Point, b: Point, c: Point) =>
  (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);

function trianglesFromConnections(
  connections: Array<{ start: number; end: number }>,
): Triangle[] {
  const triangles: Triangle[] = [];
  for (let index = 0; index + 2 < connections.length; index += 3) {
    const vertices = [
      ...new Set(
        connections
          .slice(index, index + 3)
          .flatMap((connection) => [connection.start, connection.end]),
      ),
    ];
    if (vertices.length === 3 && vertices.every((vertex) => vertex < 468)) {
      triangles.push(vertices as Triangle);
    }
  }
  return triangles;
}

/** Initialize the exact MediaPipe face topology without starting the detector. */
export function initializeFaceTopology(
  connections: Array<{ start: number; end: number }>,
) {
  canonicalFaceTriangles = trianglesFromConnections(connections);
  meshCache = new WeakMap<HTMLCanvasElement, MeshData>();
  return canonicalFaceTriangles.length;
}

function circumcircleContains(points: Point[], triangle: Triangle, point: Point) {
  const [a, b, c] = triangle.map((index) => points[index]) as [Point, Point, Point];
  const denominator =
    2 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
  if (Math.abs(denominator) < 1e-7) return false;
  const a2 = a.x * a.x + a.y * a.y;
  const b2 = b.x * b.x + b.y * b.y;
  const c2 = c.x * c.x + c.y * c.y;
  const ux =
    (a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) /
    denominator;
  const uy =
    (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) /
    denominator;
  const radiusSquared = (ux - a.x) ** 2 + (uy - a.y) ** 2;
  const distanceSquared = (ux - point.x) ** 2 + (uy - point.y) ** 2;
  return distanceSquared <= radiusSquared + 0.01;
}

function delaunay(input: Point[]): Triangle[] {
  if (input.length < 3) return [];
  const minX = Math.min(...input.map((point) => point.x));
  const maxX = Math.max(...input.map((point) => point.x));
  const minY = Math.min(...input.map((point) => point.y));
  const maxY = Math.max(...input.map((point) => point.y));
  const size = Math.max(maxX - minX, maxY - minY, 1);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const points = [
    ...input,
    { x: centerX - size * 20, y: centerY - size * 16 },
    { x: centerX, y: centerY + size * 20 },
    { x: centerX + size * 20, y: centerY - size * 16 },
  ];
  const count = input.length;
  let triangles: Triangle[] = [[count, count + 1, count + 2]];

  for (let index = 0; index < count; index += 1) {
    const bad = triangles.filter((triangle) =>
      circumcircleContains(points, triangle, points[index]),
    );
    const edges = new Map<string, { edge: [number, number]; count: number }>();
    for (const [a, b, c] of bad) {
      for (const edge of [[a, b], [b, c], [c, a]] as [number, number][]) {
        const key = edge[0] < edge[1] ? `${edge[0]}:${edge[1]}` : `${edge[1]}:${edge[0]}`;
        const current = edges.get(key);
        if (current) current.count += 1;
        else edges.set(key, { edge, count: 1 });
      }
    }
    const badSet = new Set(bad);
    triangles = triangles.filter((triangle) => !badSet.has(triangle));
    for (const { edge, count: edgeCount } of edges.values()) {
      if (edgeCount !== 1) continue;
      const triangle: Triangle = [edge[0], edge[1], index];
      if (signedArea(points[triangle[0]], points[triangle[1]], points[triangle[2]]) < 0) {
        [triangle[0], triangle[1]] = [triangle[1], triangle[0]];
      }
      triangles.push(triangle);
    }
  }

  return triangles.filter(
    (triangle) =>
      triangle.every((index) => index < count) &&
      Math.abs(signedArea(input[triangle[0]], input[triangle[1]], input[triangle[2]])) > 0.08,
  );
}

function fieldDisplacement(source: Point, controls: Control[]) {
  let sum = 0;
  let dx = 0;
  let dy = 0;
  for (const control of controls) {
    const distanceSquared =
      (source.x - control.source.x) ** 2 + (source.y - control.source.y) ** 2;
    const weight = Math.exp(-distanceSquared / (2 * control.radius ** 2));
    sum += weight;
    dx += control.dx * weight;
    dy += control.dy * weight;
  }
  if (sum === 0) return { x: 0, y: 0 };
  const influence = Math.min(1, sum);
  return {
    x: (dx / sum) * influence,
    y: (dy / sum) * influence,
  };
}

type MeshData = {
  points: Point[];
  facePointCount: number;
  triangles: Triangle[];
  boundary: Point[];
  signature: string;
};

let meshCache = new WeakMap<HTMLCanvasElement, MeshData>();

function buildFaceMesh(
  sourceCanvas: HTMLCanvasElement,
  landmarks: Point[],
): MeshData {
  const { width, height } = sourceCanvas;
  const signature = `${width}x${height}:${[10, 33, 152, 234, 263, 454]
    .map((index) => `${landmarks[index].x.toFixed(5)},${landmarks[index].y.toFixed(5)}`)
    .join(":")}`;
  const cached = meshCache.get(sourceCanvas);
  if (cached?.signature === signature) return cached;
  const facePoints = landmarks.slice(0, 468).map((point) => ({
    x: point.x * width,
    y: point.y * height,
  }));
  const points = [...facePoints];
  const facePointCount = points.length;
  const oval = FACE_OVAL.map((index) => facePoints[index]);
  const minX = Math.min(...oval.map((point) => point.x));
  const maxX = Math.max(...oval.map((point) => point.x));
  const minY = Math.min(...oval.map((point) => point.y));
  const maxY = Math.max(...oval.map((point) => point.y));
  const center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };

  // A fixed ring follows the actual face oval. Its zero-motion vertices keep the
  // hair, ears, clothing, and background unchanged outside the narrow blend band.
  const boundary: Point[] = [];
  for (const ovalPoint of oval) {
    const anchor = {
      x: clamp(center.x + (ovalPoint.x - center.x) * 1.18, 0, width),
      y: clamp(center.y + (ovalPoint.y - center.y) * 1.16, 0, height),
    };
    boundary.push(anchor);
    points.push(anchor);
  }

  const internalTriangles = canonicalFaceTriangles ?? delaunay(facePoints);
  const ringTriangles: Triangle[] = [];
  for (let index = 0; index < FACE_OVAL.length; index += 1) {
    const next = (index + 1) % FACE_OVAL.length;
    const inner = FACE_OVAL[index];
    const innerNext = FACE_OVAL[next];
    const outer = facePointCount + index;
    const outerNext = facePointCount + next;
    ringTriangles.push([inner, innerNext, outerNext], [inner, outerNext, outer]);
  }
  const triangles = [...internalTriangles, ...ringTriangles].filter(
    ([a, b, c]) => Math.abs(signedArea(points[a], points[b], points[c])) > 0.08,
  );

  const mesh = {
    points,
    facePointCount,
    triangles,
    boundary,
    signature,
  };
  meshCache.set(sourceCanvas, mesh);
  return mesh;
}

function regularizeDisplacements(displacements: Point[], mesh: MeshData) {
  const adjacency = Array.from({ length: mesh.points.length }, () => new Set<number>());
  for (const [a, b, c] of mesh.triangles) {
    adjacency[a].add(b); adjacency[a].add(c);
    adjacency[b].add(a); adjacency[b].add(c);
    adjacency[c].add(a); adjacency[c].add(b);
  }
  let current = displacements.map((point) => ({ x: point.x, y: point.y }));
  for (let iteration = 0; iteration < 2; iteration += 1) {
    current = current.map((point, index) => {
      if (index >= mesh.facePointCount || adjacency[index].size === 0) {
        return { x: 0, y: 0 };
      }
      let x = 0;
      let y = 0;
      for (const neighbor of adjacency[index]) {
        x += current[neighbor].x;
        y += current[neighbor].y;
      }
      const amount = 0.16;
      return {
        x: point.x * (1 - amount) + (x / adjacency[index].size) * amount,
        y: point.y * (1 - amount) + (y / adjacency[index].size) * amount,
      };
    });
  }
  return current;
}

function stabilizeProjectedSlivers(displacements: Point[], mesh: MeshData) {
  const slivers: Triangle[] = [];
  for (const [a, b, c] of mesh.triangles) {
    if (a >= mesh.facePointCount || b >= mesh.facePointCount || c >= mesh.facePointCount) continue;
    const points = [mesh.points[a], mesh.points[b], mesh.points[c]];
    const maximumEdge = Math.max(
      Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y),
      Math.hypot(points[1].x - points[2].x, points[1].y - points[2].y),
      Math.hypot(points[2].x - points[0].x, points[2].y - points[0].y),
      0.001,
    );
    const projectedQuality = Math.abs(signedArea(points[0], points[1], points[2])) /
      (maximumEdge * maximumEdge);
    if (projectedQuality < 0.02) slivers.push([a, b, c]);
  }

  // Average only within each local sliver. The previous transitive union could
  // connect a long chain of unrelated cheek triangles and transport one jaw
  // handle's motion across the face, creating a strain spike on the next
  // non-sliver triangle. Two local passes keep the repair spatially bounded.
  let current = displacements.map((point) => ({ ...point }));
  for (let iteration = 0; iteration < 2; iteration += 1) {
    const totals = Array.from({ length: mesh.facePointCount }, () => ({ x: 0, y: 0, count: 0 }));
    for (const [a, b, c] of slivers) {
      const average = {
        x: (current[a].x + current[b].x + current[c].x) / 3,
        y: (current[a].y + current[b].y + current[c].y) / 3,
      };
      for (const index of [a, b, c]) {
        totals[index].x += average.x;
        totals[index].y += average.y;
        totals[index].count += 1;
      }
    }
    current = current.map((point, index) => {
      const total = totals[index];
      if (!total?.count) return point;
      return {
        x: point.x * 0.15 + (total.x / total.count) * 0.85,
        y: point.y * 0.15 + (total.y / total.count) * 0.85,
      };
    });
  }
  return current;
}

function stabilizeProjectedSliverGroups(displacements: Point[], mesh: MeshData) {
  const parent = Array.from({ length: mesh.facePointCount }, (_, index) => index);
  const involved = new Set<number>();
  const find = (index: number): number => {
    if (parent[index] !== index) parent[index] = find(parent[index]);
    return parent[index];
  };
  const join = (a: number, b: number) => {
    const rootA = find(a);
    const rootB = find(b);
    if (rootA !== rootB) parent[rootB] = rootA;
  };
  for (const [a, b, c] of mesh.triangles) {
    if (a >= mesh.facePointCount || b >= mesh.facePointCount || c >= mesh.facePointCount) continue;
    const points = [mesh.points[a], mesh.points[b], mesh.points[c]];
    const maximumEdge = Math.max(
      Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y),
      Math.hypot(points[1].x - points[2].x, points[1].y - points[2].y),
      Math.hypot(points[2].x - points[0].x, points[2].y - points[0].y),
      0.001,
    );
    const projectedQuality = Math.abs(signedArea(points[0], points[1], points[2])) /
      (maximumEdge * maximumEdge);
    if (projectedQuality < 0.012) {
      join(a, b);
      join(a, c);
      involved.add(a);
      involved.add(b);
      involved.add(c);
    }
  }
  const groups = new Map<number, number[]>();
  for (const index of involved) {
    const root = find(index);
    const group = groups.get(root) ?? [];
    group.push(index);
    groups.set(root, group);
  }
  const stabilized = displacements.map((point) => ({ ...point }));
  for (const indices of groups.values()) {
    if (indices.length < 2) continue;
    const average = indices.reduce(
      (sum, index) => ({
        x: sum.x + displacements[index].x / indices.length,
        y: sum.y + displacements[index].y / indices.length,
      }),
      { x: 0, y: 0 },
    );
    for (const index of indices) stabilized[index] = average;
  }
  return stabilized;
}

function planIsSafe(
  source: Point[],
  target: Point[],
  triangles: Triangle[],
  width: number,
  height: number,
  strictProjectedSlivers = false,
) {
  if (target.some((point) =>
    !Number.isFinite(point.x) ||
    !Number.isFinite(point.y) ||
    point.x < 0 ||
    point.y < 0 ||
    point.x > width ||
    point.y > height
  )) {
    return false;
  }
  const edgeRatio = (a: number, b: number) => {
    const sourceLength = Math.hypot(source[a].x - source[b].x, source[a].y - source[b].y);
    if (sourceLength < 1.5) return 1;
    const targetLength = Math.hypot(target[a].x - target[b].x, target[a].y - target[b].y);
    return targetLength / sourceLength;
  };
  return triangles.every(([a, b, c]) => {
    const sourceArea = signedArea(source[a], source[b], source[c]);
    const targetArea = signedArea(target[a], target[b], target[c]);
    const areaRatio = Math.abs(targetArea / sourceArea);
    // A foldover is never negotiable. Ratio checks, however, are numerically
    // noisy on sub-pixel mesh slivers and used to make one tiny triangle erase
    // an otherwise safe whole-face edit.
    if (sourceArea * targetArea <= 0) return false;
    const sourceEdges = [[a, b], [b, c], [c, a]].map(([start, end]) =>
      Math.hypot(source[start].x - source[end].x, source[start].y - source[end].y),
    );
    const minimumImageDimension = Math.min(width, height);
    const normalizedArea = Math.abs(sourceArea) / Math.max(width * height, 1);
    const normalizedMinimumEdge = Math.min(...sourceEdges) / Math.max(minimumImageDimension, 1);
    const projectedSliver = normalizedArea < 0.000001 || normalizedMinimumEdge < 0.0015;
    if (projectedSliver) {
      // Projected slivers can be only a fraction of a pixel wide even on a
      // clean frontal face. Judge their neighboring motion at a photographic
      // scale so a smooth contour edit is not capped by one degenerate edge.
      const differentialLimit = clamp(minimumImageDimension * 0.0024, 0.35, 2.8);
      const safe = [[a, b], [b, c], [c, a]].every(([start, end]) => {
        const startDx = target[start].x - source[start].x;
        const startDy = target[start].y - source[start].y;
        const endDx = target[end].x - source[end].x;
        const endDy = target[end].y - source[end].y;
        return Math.hypot(startDx - endDx, startDy - endDy) <= differentialLimit;
      });
      if (!safe) return false;
      if (!strictProjectedSlivers) return true;
    } else {
      if (areaRatio < 0.54 || areaRatio > 1.68) return false;
      const edgesSafe = [[a, b], [b, c], [c, a]].every(([start, end]) => {
        const ratio = edgeRatio(start, end);
        return ratio > 0.68 && ratio < 1.45;
      });
      if (!edgesSafe) return false;
    }

    const sx1 = source[b].x - source[a].x;
    const sy1 = source[b].y - source[a].y;
    const sx2 = source[c].x - source[a].x;
    const sy2 = source[c].y - source[a].y;
    const tx1 = target[b].x - target[a].x;
    const ty1 = target[b].y - target[a].y;
    const tx2 = target[c].x - target[a].x;
    const ty2 = target[c].y - target[a].y;
    const determinant = sx1 * sy2 - sx2 * sy1;
    const f00 = (tx1 * sy2 - tx2 * sy1) / determinant;
    const f01 = (-tx1 * sx2 + tx2 * sx1) / determinant;
    const f10 = (ty1 * sy2 - ty2 * sy1) / determinant;
    const f11 = (-ty1 * sx2 + ty2 * sx1) / determinant;
    const frobeniusSquared = f00 ** 2 + f01 ** 2 + f10 ** 2 + f11 ** 2;
    const determinantSquared = (f00 * f11 - f01 * f10) ** 2;
    const discriminant = Math.sqrt(
      Math.max(0, frobeniusSquared ** 2 - 4 * determinantSquared),
    );
    const maximumStretch = Math.sqrt((frobeniusSquared + discriminant) / 2);
    const minimumStretch = Math.sqrt(
      Math.max(0, (frobeniusSquared - discriminant) / 2),
    );
    return (
      minimumStretch >= 0.68 &&
      maximumStretch < 1.48 &&
      maximumStretch / Math.max(minimumStretch, 0.001) < 1.8
    );
  });
}

export function morphImage(
  sourceCanvas: HTMLCanvasElement,
  landmarks: Point[],
  plan: MorphPlan,
  strengthPercent: number,
): MorphResult {
  const { width, height } = sourceCanvas;
  const output = document.createElement("canvas");
  output.width = width;
  output.height = height;
  const ctx = output.getContext("2d", { alpha: false });
  if (!ctx) throw new Error("Canvas rendering is unavailable in this browser.");

  const requestedStrength = Math.max(0, Math.min(1, strengthPercent / 100));
  if (requestedStrength === 0) {
    ctx.drawImage(sourceCanvas, 0, 0);
    return {
      canvas: output,
      movedRegions: [],
      maxMovementPx: 0,
      safetyScale: 0,
      perRegionScale: {},
      safetyStatus: "identity",
      plan,
    };
  }

  const frame = createPoseFrame(landmarks, width, height);
  const projectedFaceWidth = Math.max(
    Math.abs(frame.local(454).u - frame.local(234).u),
    1,
  );
  const projectedFaceHeight = Math.max(
    Math.abs(frame.local(152).v - frame.local(10).v),
    1,
  );
  const noseU = frame.local(1).u;
  const noseToLeft = Math.abs(noseU - frame.local(234).u);
  const noseToRight = Math.abs(frame.local(454).u - noseU);
  const morphYawSignal = Math.abs(noseToLeft - noseToRight) /
    Math.max(noseToLeft + noseToRight, 1);
  const isProfilePlan = (
    plan.evidenceFamilies?.some((family) => family.id.includes(".profile-")) ?? false
  ) || projectedFaceWidth / Math.max(projectedFaceHeight * 0.65, 1) < 0.62;
  const smoothUnit = (value: number) => {
    const t = clamp(value, 0, 1);
    return t * t * (3 - 2 * t);
  };
  // Keep the visible control honest: 100% now requests the full bounded plan.
  // Mild yaw retains a small reserve for projected slivers, while the regional
  // binary-search validator remains the final authority on safe amplitude.
  const mildYawReserve = isProfilePlan
    ? 0
    : smoothUnit((morphYawSignal - 0.08) / 0.1) *
      (1 - smoothUnit((morphYawSignal - 0.5) / 0.15));
  const calibratedMaximum = 0.8 - mildYawReserve * 0.25;
  const calibratedStrength = requestedStrength * calibratedMaximum;
  const strength = calibratedStrength ** 2.5;
  const controls = buildControls(landmarks, width, height, plan, strength);
  const mesh = buildFaceMesh(sourceCanvas, landmarks);
  const isNearFrontalPlan = !isProfilePlan && morphYawSignal < 0.18;
  const faceScale = Math.max(projectedFaceWidth, projectedFaceHeight * 0.65);
  const wholeFaceMovementLimit = faceScale * 0.04;
  const jawAction = plan.actions.find((action) => action.region === "Jaw");
  const requestedJawCap = Number(jawAction?.maxDisplacement);
  const jawCap = Number.isFinite(requestedJawCap)
    ? clamp(requestedJawCap, 0, 0.045)
    : 0.045;
  const jawPressure = clamp(jawAction?.amount ?? 0, -0.9, 0.9) * faceScale * 0.18 * strength;
  const jawMagnitude = clamp(jawPressure, -faceScale * jawCap, faceScale * jawCap);
  const jawStartV = frame.local(13).v - projectedFaceHeight * 0.08;
  const jawFullV = frame.local(152).v - projectedFaceHeight * 0.06;
  const halfFaceWidth = projectedFaceWidth / 2;
  const smoothstep = (start: number, end: number, value: number) => {
    const t = clamp((value - start) / Math.max(end - start, 0.001), 0, 1);
    return t * t * (3 - 2 * t);
  };
  const frontalJawDisplacement = (point: Point) => {
    const local = frame.localPoint(point);
    const verticalWeight = smoothstep(
      jawStartV - projectedFaceHeight * 0.04,
      jawFullV - projectedFaceHeight * 0.08,
      local.v,
    );
    const sideWeight = smoothstep(
      halfFaceWidth * 0.08,
      halfFaceWidth * 0.5,
      Math.abs(local.u),
    );
    const horizontal = Math.sign(local.u || 1) * jawMagnitude * verticalWeight * sideWeight;
    const displacement = frame.delta(horizontal, 0);
    return { x: displacement.dx, y: displacement.dy };
  };
  const controlsByRegion = new Map<string, Control[]>();
  for (const control of controls) {
    const regionControls = controlsByRegion.get(control.region) ?? [];
    regionControls.push(control);
    controlsByRegion.set(control.region, regionControls);
  }
  const regionOrder = [...new Set([
    ...plan.actions.map((action) => action.region),
    ...controlsByRegion.keys(),
  ])];
  let targetPoints = mesh.points.map((point) => ({ ...point }));
  const perRegionScale: Record<string, number> = {};
  const movedRegions: string[] = [];
  let weightedScale = 0;
  let weightTotal = 0;
  let anyBackoff = false;

  // Admit each region independently against the already-safe target. A local
  // nose or lip constraint can now weaken that region without shrinking the
  // jaw, chin, and brow edits that already passed.
  for (const region of regionOrder) {
    const regionControls = controlsByRegion.get(region) ?? [];
    if (!regionControls.length) continue;
    const legacyRawDisplacements = mesh.points.map((point, index) =>
      index < mesh.facePointCount
        ? fieldDisplacement(point, regionControls)
        : { x: 0, y: 0 },
    );
    const legacySmoothed = regularizeDisplacements(legacyRawDisplacements, mesh);
    const legacyUsesLocalSlivers = isNearFrontalPlan;
    const displacementCandidates: Array<{
      displacements: Point[];
      strictProjectedSlivers: boolean;
    }> = [{
      displacements: legacyUsesLocalSlivers
        ? stabilizeProjectedSlivers(legacySmoothed, mesh)
        : stabilizeProjectedSliverGroups(legacySmoothed, mesh),
      strictProjectedSlivers: legacyUsesLocalSlivers,
    }];

    if (region === "Jaw" && !isProfilePlan && jawAction) {
      const analyticRaw = mesh.points.map((point, index) =>
        index < mesh.facePointCount ? frontalJawDisplacement(point) : { x: 0, y: 0 },
      );
      displacementCandidates.push({
        displacements: stabilizeProjectedSlivers(analyticRaw, mesh),
        // This is a smooth whole-lower-face field rather than independent
        // local handles. Keep foldover and differential checks, but do not
        // apply unstable affine-stretch tests to near-zero projected slivers.
        strictProjectedSlivers: true,
      });
    }

    const actionPeak = Math.max(
      ...plan.actions
        .filter((action) => action.region === region)
        .map((action) => Math.abs(action.amount)),
      0,
    );
    // Planned primitives are contractually bounded to ±0.9. Preserve a safe,
    // visible portion of malformed or future oversized requests instead of
    // trusting the control clamp and reporting a false pass.
    const contractScale = actionPeak > 0.9 ? 0.98 : 1;
    let sawNonFiniteCandidate = false;
    const evaluatedCandidates = displacementCandidates.flatMap((candidate) => {
      const regionWeight = Math.max(
        ...candidate.displacements.map((point) => Math.hypot(point.x, point.y)),
        0,
      );
      if (!Number.isFinite(regionWeight)) {
        sawNonFiniteCandidate = true;
        return [];
      }
      if (regionWeight <= 0.01) return [];

      const candidateFor = (scaleAmount: number) => targetPoints.map((point, index) => {
        const next = {
          x: point.x + candidate.displacements[index].x * scaleAmount,
          y: point.y + candidate.displacements[index].y * scaleAmount,
        };
        if (index >= mesh.facePointCount) return next;
        const source = mesh.points[index];
        const dx = next.x - source.x;
        const dy = next.y - source.y;
        const magnitude = Math.hypot(dx, dy);
        if (magnitude <= wholeFaceMovementLimit) return next;
        const boundedScale = wholeFaceMovementLimit / Math.max(magnitude, 0.001);
        return {
          x: source.x + dx * boundedScale,
          y: source.y + dy * boundedScale,
        };
      });

      let regionScale = contractScale;
      if (!planIsSafe(
        mesh.points,
        candidateFor(regionScale),
        mesh.triangles,
        width,
        height,
        candidate.strictProjectedSlivers,
      )) {
        let safeScale = 0;
        let unsafeScale = regionScale;
        for (let attempt = 0; attempt < 11; attempt += 1) {
          const candidateScale = (safeScale + unsafeScale) / 2;
          if (planIsSafe(
            mesh.points,
            candidateFor(candidateScale),
            mesh.triangles,
            width,
            height,
            candidate.strictProjectedSlivers,
          )) {
            safeScale = candidateScale;
          } else {
            unsafeScale = candidateScale;
          }
        }
        regionScale = safeScale;
      }
      if (regionScale < 0.02) regionScale = 0;
      const acceptedTarget = candidateFor(regionScale);
      const displacementMagnitudes = acceptedTarget
        .slice(0, mesh.facePointCount)
        .map((point, index) => Math.hypot(
          point.x - targetPoints[index].x,
          point.y - targetPoints[index].y,
        ));
      const sortedMagnitudes = [...displacementMagnitudes].sort((a, b) => a - b);
      const p95Impact = sortedMagnitudes[
        Math.floor(Math.max(sortedMagnitudes.length - 1, 0) * 0.95)
      ] ?? 0;
      const meanImpact = displacementMagnitudes.reduce((total, value) => total + value, 0) /
        Math.max(displacementMagnitudes.length, 1);
      const impactScore = p95Impact + meanImpact * 0.1;
      return [{ acceptedTarget, impactScore, regionScale, regionWeight }];
    });

    if (!evaluatedCandidates.length) {
      perRegionScale[region] = 0;
      if (sawNonFiniteCandidate) anyBackoff = true;
      continue;
    }
    const selected = region === "Jaw" && isNearFrontalPlan
      ? evaluatedCandidates.at(-1)!
      : evaluatedCandidates.reduce((best, candidate) =>
          candidate.impactScore > best.impactScore ? candidate : best,
        );
    perRegionScale[region] = selected.regionScale;
    weightedScale += selected.regionScale * selected.regionWeight;
    weightTotal += selected.regionWeight;
    if (selected.regionScale < 0.995) anyBackoff = true;
    if (selected.regionScale === 0) continue;
    targetPoints = selected.acceptedTarget;
    movedRegions.push(region);
  }

  const safetyScale = weightTotal > 0 ? weightedScale / weightTotal : 0;
  const maxMovementPx = Math.max(
    ...targetPoints.map((point, index) => Math.hypot(
      point.x - mesh.points[index].x,
      point.y - mesh.points[index].y,
    )),
    0,
  );

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(sourceCanvas, 0, 0);
  const identityOnly = controls.length === 0 || maxMovementPx < 0.05 || mesh.triangles.length === 0;
  if (!identityOnly) {
    const layer = document.createElement("canvas");
    layer.width = width;
    layer.height = height;
    const layerContext = layer.getContext("2d");
    if (!layerContext) throw new Error("Canvas rendering is unavailable in this browser.");
    layerContext.imageSmoothingEnabled = true;
    layerContext.imageSmoothingQuality = "high";
    // Give the After image an immediately legible finish in addition to the
    // structural warp. Restrict it to the anchored face layer so hair,
    // clothing, and background remain unchanged.
    // Render the complete anchored face region once. Drawing only moved triangles
    // over the original would leave duplicate contours when a feature contracts.
    layerContext.save();
    layerContext.beginPath();
    mesh.boundary.forEach((point, index) => {
      if (index === 0) layerContext.moveTo(point.x, point.y);
      else layerContext.lineTo(point.x, point.y);
    });
    layerContext.closePath();
    layerContext.clip();
    for (const [a, b, c] of mesh.triangles) {
      drawTriangle(
        layerContext,
        sourceCanvas,
        [mesh.points[a], mesh.points[b], mesh.points[c]],
        [targetPoints[a], targetPoints[b], targetPoints[c]],
      );
    }
    layerContext.restore();
    ctx.drawImage(layer, 0, 0);

    // Add a feathered face-only finish after geometry is rendered. The exact
    // face boundary remains the hard outer clip, while the soft inner mask
    // avoids a visible tonal seam across the forehead or cheeks.
    const finish = document.createElement("canvas");
    finish.width = width;
    finish.height = height;
    const finishContext = finish.getContext("2d");
    if (finishContext && typeof finishContext.createRadialGradient === "function") {
      const editorialIntensity = clamp(plan.editorialIntensity ?? 0);
      const finishStrength = requestedStrength * (0.3 + editorialIntensity * 0.7);
      finishContext.filter = [
        `contrast(${1 + finishStrength * 0.22})`,
        `saturate(${1 + finishStrength * 0.14})`,
        `brightness(${1 + finishStrength * 0.035})`,
      ].join(" ");
      finishContext.drawImage(output, 0, 0);
      finishContext.filter = "none";
      if (editorialIntensity > 0.01) {
        // A light low-frequency blend softens small distractions without
        // erasing the original skin texture or redrawing any anatomy.
        finishContext.globalAlpha = editorialIntensity * 0.16;
        finishContext.filter = `blur(${(editorialIntensity * 1.15).toFixed(2)}px) brightness(1.025)`;
        finishContext.drawImage(output, 0, 0);
        finishContext.globalAlpha = 1;
        finishContext.filter = "none";
      }
      const mask = document.createElement("canvas");
      mask.width = width;
      mask.height = height;
      const maskContext = mask.getContext("2d");
      if (!maskContext) throw new Error("Canvas rendering is unavailable in this browser.");
      maskContext.beginPath();
      mesh.boundary.forEach((point, index) => {
        if (index === 0) maskContext.moveTo(point.x, point.y);
        else maskContext.lineTo(point.x, point.y);
      });
      maskContext.closePath();
      maskContext.clip();
      const xs = mesh.boundary.map((point) => point.x);
      const ys = mesh.boundary.map((point) => point.y);
      const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
      const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
      const radiusX = Math.max((Math.max(...xs) - Math.min(...xs)) / 2, 1);
      const radiusY = Math.max((Math.max(...ys) - Math.min(...ys)) / 2, 1);
      maskContext.save();
      maskContext.translate(centerX, centerY);
      maskContext.scale(1, radiusY / radiusX);
      const gradient = maskContext.createRadialGradient(0, 0, 0, 0, 0, radiusX);
      gradient.addColorStop(0, "rgba(255,255,255,1)");
      gradient.addColorStop(0.72, "rgba(255,255,255,0.96)");
      gradient.addColorStop(1, "rgba(255,255,255,0)");
      maskContext.fillStyle = gradient;
      maskContext.fillRect(-radiusX, -radiusX, radiusX * 2, radiusX * 2);
      maskContext.restore();
      finishContext.globalCompositeOperation = "destination-in";
      finishContext.drawImage(mask, 0, 0);
      ctx.drawImage(finish, 0, 0);
    }
  }

  return {
    canvas: output,
    movedRegions: identityOnly ? [] : movedRegions,
    maxMovementPx: identityOnly ? 0 : maxMovementPx,
    safetyScale: identityOnly ? 0 : safetyScale,
    perRegionScale,
    safetyStatus: identityOnly ? "identity" : anyBackoff ? "weakened" : "passed",
    plan,
  };
}

export function drawLandmarkOverlay(
  canvas: HTMLCanvasElement,
  landmarks: Point[],
  color = "#c8ff45",
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.78;
  for (const index of semanticOverlayIndices(landmarks.length)) {
    const point = landmarks[index];
    ctx.beginPath();
    ctx.arc(point.x * canvas.width, point.y * canvas.height, Math.max(1.15, canvas.width / 820), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
