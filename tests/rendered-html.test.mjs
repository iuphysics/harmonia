import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

async function readSources() {
  const [engine, intelligence, page, specification] = await Promise.all([
    readFile(new URL("../lib/morph.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/face-intelligence.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../docs/facial-analysis-spec.md", import.meta.url), "utf8"),
  ]);
  return { engine, intelligence, page, specification };
}

test("renders the V2 blended Harmonia studio", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();

  assert.match(html, /<title>Harmonia — Pixel-only portrait morphing<\/title>/i);
  assert.match(html, /Scan your face/);
  assert.match(html, /Adjust your look/);
  assert.match(html, /Refine direction weight/);
  assert.match(html, /Balance direction weight/);
  assert.match(html, /Definition direction weight/);
  assert.match(html, /One personalized plan/);
  assert.match(html, /Overall change/);
  assert.match(html, /Strong/);
  assert.match(html, /Start camera/);
  assert.match(html, /Export exact PNG/);
  assert.match(html, /Never uploaded/);
  assert.match(html, /HARMONIA \/ V2/);
  assert.doesNotMatch(html, /Chadlite|objective beauty|universal ideal/i);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("ships the local detector and fixed-topology pixel warp", async () => {
  const { engine, page } = await readSources();

  await Promise.all([
    access(new URL("../public/models/face_landmarker.task", import.meta.url)),
    access(new URL("../public/mediapipe/wasm/vision_wasm_internal.wasm", import.meta.url)),
  ]);

  assert.match(engine, /FaceLandmarker/);
  assert.match(
    engine,
    /FilesetResolver\.forVisionTasks\(\s*publicAsset\("mediapipe\/wasm"\)/,
  );
  assert.match(engine, /outputFaceBlendshapes:\s*true/);
  assert.match(engine, /outputFacialTransformationMatrixes:\s*true/);
  assert.match(engine, /detectFace/);
  assert.match(engine, /drawTriangle/);
  assert.match(engine, /buildFaceMesh/);
  assert.match(engine, /FACE_LANDMARKS_TESSELATION/);
  assert.match(engine, /regularizeDisplacements/);
  assert.match(engine, /fixed ring follows the actual face oval/i);
  assert.match(engine, /planIsSafe/);
  assert.match(engine, /sourceArea \* targetArea <= 0/);
  assert.match(engine, /maximumStretch \/ Math\.max\(minimumStretch/);
  assert.match(engine, /candidateScale/);

  assert.match(page, /createInteractiveMorphPlan/);
  assert.match(page, /morphImage/);
  assert.match(page, /getUserMedia/);
  assert.match(page, /freezeMirroredFrame/);
  assert.match(page, /assessFrame/);
  assert.match(page, /laplacianSquaredTotal/);
  assert.match(page, /stableFrames >= 7/);
  assert.match(page, /sourceRef\.current = chosen\.canvas/);
  assert.doesNotMatch(page, /detectLandmarks\(video\)/);
  assert.match(page, /toDataURL\("image\/png"/);
  assert.doesNotMatch(page, /fetch\(|XMLHttpRequest|FormData/);
});

test("defines the versioned V2 facial intelligence contract", async () => {
  const { engine, intelligence, page } = await readSources();

  assert.match(intelligence, /export const SEMANTIC_LANDMARKS\b/);
  assert.match(intelligence, /export const SEMANTIC_LANDMARK_COUNT\b/);
  assert.match(intelligence, /SEMANTIC_LANDMARK_COUNT[\s\S]{0,160}\b104\b|104[\s\S]{0,160}SEMANTIC_LANDMARK_COUNT/);
  assert.match(intelligence, /export const MEASUREMENT_CATALOG\b/);
  assert.match(intelligence, /MEASUREMENT_CATALOG[\s\S]{0,300}\b408\b|\b408\b[\s\S]{0,300}MEASUREMENT_CATALOG/);
  assert.match(intelligence, /export const DEFAULT_DIRECTION_MIX\b/);
  assert.match(intelligence, /export (?:function|const) analyzeFace\b/);
  assert.match(intelligence, /export (?:function|const) createMorphPlan\b/);

  assert.match(intelligence, /frontal/);
  assert.match(intelligence, /three-quarter/);
  assert.match(intelligence, /profile/);
  assert.match(intelligence, /confidence/i);
  assert.match(intelligence, /editability/i);
  assert.match(intelligence, /plannerUse/);
  assert.match(intelligence, /guardrail/);
  assert.match(intelligence, /preserve/);
  assert.match(intelligence, /no[- ]op|identity/i);

  assert.match(engine, /SEMANTIC_LANDMARKS|SEMANTIC_LANDMARK_COUNT/);
  assert.match(engine, /MEASUREMENT_CATALOG/);
  assert.match(engine, /DEFAULT_DIRECTION_MIX/);

  assert.match(page, /type DirectionMix/);
  assert.match(page, /directionMix/);
  assert.match(page, /Adjust your look/);
  assert.match(page, /semantic anchors/);
  assert.match(page, /derived measures/);
  assert.match(page, /valid for pose/);
  assert.match(page, /Region editability/);
  assert.match(page, /Preserve this geometry/);
  assert.doesNotMatch(page, /presetId|setPresetId|PRESETS\.map/);
});

test("documents the exact counts, pose gates, planner, and limitations", async () => {
  const { specification } = await readSources();

  assert.match(specification, /Semantic proxy count:\s*\*\*104\*\*/);
  assert.match(specification, /Measurement catalog count:\s*\*\*408\*\*/);
  assert.match(specification, /Distance and span\s*\|\s*78/);
  assert.match(specification, /Ratio\s*\|\s*210/);
  assert.match(specification, /Angle\s*\|\s*36/);
  assert.match(specification, /Shape\s*\|\s*20/);
  assert.match(specification, /Symmetry\s*\|\s*40/);
  assert.match(specification, /Context\s*\|\s*24/);
  assert.match(specification, /not[\s\S]{0,120}MediaPipe[\s\S]{0,120}anatomical landmark/i);
  assert.match(specification, /bilateral symmetry[\s\S]{0,120}\|\s*no\s*\|/i);
  assert.match(specification, /measurement confidence\s*=/i);
  assert.match(specification, /Region editability/);
  assert.match(specification, /Joint planner/);
  assert.match(specification, /conservative no-op|produce a no-op/i);
  assert.match(specification, /no dependable hairline or ear coverage/i);
  assert.match(specification, /not metric millimetres/i);
  assert.match(specification, /do not measure true bone projection/i);
  assert.match(specification, /cannot create anatomy, texture, lighting, shadows, hair, skin/i);
  assert.match(specification, /MediaPipe Face Mesh/);
  assert.match(specification, /Face Alignment Across Large Poses/);
  assert.match(specification, /3D Facial Norms/);
});
