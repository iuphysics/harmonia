import {
  PLANNER_SCHEMA_VERSION,
  buildPlannerEvidence,
  type PlannerEvidenceFamily,
} from "./planner-evidence.ts";

export { PLANNER_RULES, PLANNER_SCHEMA_VERSION } from "./planner-evidence.ts";
export type { PlannerEvidenceFamily } from "./planner-evidence.ts";

/**
 * Harmonia's pose-aware analysis and planning layer.
 *
 * MediaPipe supplies a dense, unnamed mesh. The names below are Harmonia V2
 * soft-tissue proxies layered over that topology; they are not clinical
 * landmarks and are not official names published by MediaPipe.
 */

export type Point = { x: number; y: number; z?: number };

export type FaceObservation = {
  landmarks: Point[];
  blendshapes: Record<string, number>;
  transformationMatrix?: { rows: number; columns: number; data: number[] };
};

export type PoseClass =
  | "frontal"
  | "three-quarter-left"
  | "three-quarter-right"
  | "profile-left"
  | "profile-right"
  | "unsupported";

export type PoseEstimate = {
  class: PoseClass;
  label: string;
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  confidence: number;
  visibleSide: "left" | "right" | "both";
};

export type DirectionMix = {
  harmony: number;
  symmetry: number;
  dimorphism: number;
};

export type FaceRegion = "Jaw" | "Chin" | "Nose" | "Lips" | "Brows" | "Face shape" | "Symmetry";

export type RegionEditability = {
  region: FaceRegion;
  score: number;
  editable: boolean;
  reasons: string[];
  maxDisplacement: number;
};

export type MeasurementKind = "distance" | "ratio" | "angle" | "shape" | "symmetry" | "context";
export type PlannerUse = "target" | "guardrail" | "confidence" | "context" | "preserve";
export type PoseSupport = "all" | "bilateral" | "frontal" | "visible-side" | "profile";

export type MeasurementDefinition = {
  id: string;
  label: string;
  kind: MeasurementKind;
  region: string;
  inputs: string[];
  poseSupport: PoseSupport;
  validPoses: PoseClass[];
  plannerUse: PlannerUse;
  consumerRules: string[];
  unit: "pixels" | "ratio" | "degrees" | "normalized";
  operation?: "distance" | "ratio" | "angle" | "polyline" | "curvature" | "area" | "axial" | "vertical";
};

export type MeasurementValue = {
  id: string;
  label: string;
  kind: MeasurementKind;
  region: string;
  value: number | null;
  confidence: number;
  editability: number;
  validity: "valid" | "degraded" | "invalid";
  reasons: string[];
  plannerUse: PlannerUse;
};

export type FaceAnalysis = {
  pose: PoseEstimate;
  semanticLandmarkCount: number;
  measurementCount: number;
  validMeasurementCount: number;
  overallConfidence: number;
  expression: {
    label: string;
    confidence: number;
    blockedRegions: string[];
  };
  regionEditability: RegionEditability[];
  measurements: MeasurementValue[];
  unavailableMeasurements: string[];
  metrics: {
    jawToFace: number;
    noseToFace: number;
    mouthToFace: number;
    lowerThird: number;
    pairedDeviation: number;
    mouthToNose: number;
    chinToJaw: number;
    faceAspect: number;
  };
};

export type MorphPrimitive =
  | "jaw-width"
  | "chin-length"
  | "nose-width"
  | "mouth-width"
  | "brow-height"
  | "paired-alignment";

export type PlannedAction = {
  primitive: MorphPrimitive;
  region: FaceRegion;
  amount: number;
  confidence: number;
  editability: number;
  /** Maximum handle displacement as a fraction of the pose-aware control scale. */
  maxDisplacement: number;
  directions: Array<keyof DirectionMix>;
  rationale: string;
};

export type MorphPlan = {
  schemaVersion: typeof PLANNER_SCHEMA_VERSION;
  actions: PlannedAction[];
  evidenceFamilies: PlannerEvidenceFamily[];
  evidenceMeasurementCount: number;
  preservedRegions: string[];
  rejectedReasons: string[];
  candidateCount: number;
  selectedCandidate: "identity" | "light" | "balanced" | "full";
  directionContributions: DirectionMix;
  /** Coordinated high-fashion finish unlocked when all editor directions are high. */
  editorialIntensity?: number;
};

export type AnalysisOptions = {
  qualityConfidence?: number;
  temporalStability?: number;
  resolutionSupport?: number;
  exposureSupport?: number;
};

export type SemanticLandmarkDefinition = {
  id: string;
  region: string;
  side: "midline" | "right" | "left";
  meshIndex: number;
  fallbackIndices?: number[];
  reliability: number;
  kind: "surface" | "soft-tissue-proxy" | "curve-sample";
};

export const DEFAULT_DIRECTION_MIX: DirectionMix = {
  harmony: 70,
  symmetry: 25,
  dimorphism: 55,
};

const clamp = (value: number, min = 0, max = 1) => Math.max(min, Math.min(max, value));
const poses = (...values: PoseClass[]) => values;
const ALL_SUPPORTED_POSES = poses("frontal", "three-quarter-left", "three-quarter-right", "profile-left", "profile-right");

const MIDLINE: Record<string, number> = {
  "forehead.top": 10,
  "forehead.center": 151,
  "forehead.glabellaProxy": 9,
  "nose.nasionProxy": 168,
  "nose.bridgeUpper": 6,
  "nose.bridgeMid": 197,
  "nose.bridgeLower": 5,
  "nose.pronasaleProxy": 1,
  "nose.subnasaleProxy": 2,
  "mouth.labraleSuperius": 0,
  "mouth.stomionUpper": 13,
  "mouth.stomionLower": 14,
  "mouth.labraleInferius": 17,
  "chin.menton": 152,
};

/** Tuple order is the subject's anatomical right, then anatomical left. */
const BILATERAL: Record<string, [number, number]> = {
  "outline.frontotemporaleProxy": [103, 332],
  "outline.templeHigh": [54, 284],
  "outline.templeLow": [127, 356],
  "outline.zygionProxy": [234, 454],
  "outline.cheek": [132, 361],
  "outline.gonionProxy": [172, 397],
  "outline.jawAngleLower": [136, 365],
  "outline.jawMid": [150, 379],
  "outline.preChin": [176, 400],
  "outline.chinLateral": [148, 377],
  "brow.innerTop": [107, 336],
  "brow.innerBottom": [66, 296],
  "brow.midBottom": [105, 334],
  "brow.archBottom": [63, 293],
  "brow.outerBottom": [70, 300],
  "brow.archTop": [52, 282],
  "eye.outerCanthus": [33, 263],
  "eye.innerCanthus": [133, 362],
  "eye.upperOuter": [160, 387],
  "eye.upperApex": [159, 386],
  "eye.upperInner": [158, 385],
  "eye.lowerInner": [155, 382],
  "eye.lowerApex": [153, 380],
  "eye.lowerOuter": [144, 373],
  "eye.irisCenter": [468, 473],
  "nose.alarWidest": [98, 327],
  "nose.alarBase": [97, 326],
  "nose.alarLower": [64, 294],
  "nose.nostrilLateral": [48, 278],
  "nose.sideLower": [115, 344],
  "nose.sideMid": [220, 440],
  "nose.sideUpper": [45, 275],
  "mouth.commissure": [61, 291],
  "mouth.upperOuterLateral": [185, 409],
  "mouth.upperOuterMid": [40, 270],
  "mouth.cupidPeak": [37, 267],
  "mouth.lowerOuterLateral": [146, 375],
  "mouth.lowerOuterMid": [91, 321],
  "mouth.innerCommissure": [78, 308],
  "mouth.innerUpper": [80, 310],
  "mouth.innerLower": [88, 318],
  "mouth.lowerOuterMedial": [84, 314],
  "cheek.malarHigh": [117, 346],
  "cheek.malarLateral": [123, 352],
  "cheek.medial": [205, 425],
};

const semanticRegion = (id: string) => id.split(".")[0];

export const SEMANTIC_LANDMARKS: SemanticLandmarkDefinition[] = [
  ...Object.entries(MIDLINE).map(([id, meshIndex]) => ({
    id,
    region: semanticRegion(id),
    side: "midline" as const,
    meshIndex,
    reliability: id.includes("Proxy") ? 0.82 : 0.9,
    kind: id.includes("Proxy") ? "soft-tissue-proxy" as const : "surface" as const,
  })),
  ...Object.entries(BILATERAL).flatMap(([role, [right, left]]) => [
    {
      id: `${role}.right`,
      region: semanticRegion(role),
      side: "right" as const,
      meshIndex: right,
      fallbackIndices: role === "eye.irisCenter" ? [159, 145, 33, 133] : undefined,
      reliability: role === "eye.irisCenter" ? 0.72 : role.includes("Proxy") ? 0.8 : 0.88,
      kind: role.includes("outline") ? "curve-sample" as const : role.includes("Proxy") ? "soft-tissue-proxy" as const : "surface" as const,
    },
    {
      id: `${role}.left`,
      region: semanticRegion(role),
      side: "left" as const,
      meshIndex: left,
      fallbackIndices: role === "eye.irisCenter" ? [386, 374, 263, 362] : undefined,
      reliability: role === "eye.irisCenter" ? 0.72 : role.includes("Proxy") ? 0.8 : 0.88,
      kind: role.includes("outline") ? "curve-sample" as const : role.includes("Proxy") ? "soft-tissue-proxy" as const : "surface" as const,
    },
  ]),
];

export const SEMANTIC_LANDMARK_COUNT = SEMANTIC_LANDMARKS.length;
if (SEMANTIC_LANDMARK_COUNT !== 104) {
  throw new Error(`Harmonia semantic schema expected 104 proxies, received ${SEMANTIC_LANDMARK_COUNT}.`);
}

type DistanceSeed = {
  id: string;
  label: string;
  a: string;
  b: string;
  region: string;
  support: PoseSupport;
  use?: PlannerUse;
};

const midlineDistanceSeeds: DistanceSeed[] = [
  ["face_height", "Total face height", "forehead.top", "chin.menton", "frame"],
  ["upper_face", "Upper face height", "forehead.top", "nose.nasionProxy", "frame"],
  ["forehead", "Forehead proxy height", "forehead.top", "forehead.glabellaProxy", "frame"],
  ["upper_bridge", "Upper bridge length", "forehead.glabellaProxy", "nose.bridgeUpper", "nose"],
  ["bridge", "Bridge length", "nose.nasionProxy", "nose.pronasaleProxy", "nose"],
  ["nose_height", "Nose height", "nose.nasionProxy", "nose.subnasaleProxy", "nose"],
  ["tip_to_base", "Tip-to-base distance", "nose.pronasaleProxy", "nose.subnasaleProxy", "nose"],
  ["philtrum", "Philtrum proxy height", "nose.subnasaleProxy", "mouth.labraleSuperius", "mouth"],
  ["upper_lip", "Upper lip height", "mouth.labraleSuperius", "mouth.stomionUpper", "mouth"],
  ["mouth_opening", "Lip separation", "mouth.stomionUpper", "mouth.stomionLower", "mouth"],
  ["lower_lip_chin", "Lower lip-to-chin height", "mouth.labraleInferius", "chin.menton", "chin"],
  ["lower_face", "Lower face height", "nose.subnasaleProxy", "chin.menton", "frame"],
].map(([id, label, a, b, region]) => ({ id: `distance.midline.${id}`, label, a, b, region, support: "all", use: region === "frame" ? "guardrail" : "target" }));

const bilateralWidthRoles = [
  "outline.frontotemporaleProxy", "outline.templeLow", "outline.zygionProxy", "outline.cheek",
  "outline.gonionProxy", "outline.jawAngleLower", "outline.jawMid", "outline.chinLateral",
  "brow.innerBottom", "brow.archBottom", "brow.outerBottom",
  "eye.outerCanthus", "eye.innerCanthus", "eye.upperApex", "eye.lowerApex", "eye.irisCenter",
  "nose.alarWidest", "nose.alarBase", "nose.nostrilLateral",
  "mouth.commissure", "mouth.cupidPeak", "mouth.innerCommissure",
  "cheek.malarHigh", "cheek.malarLateral",
];

const widthSeeds: DistanceSeed[] = bilateralWidthRoles.map((role) => ({
  id: `distance.width.${role}`,
  label: `${role.replaceAll(".", " ")} width`,
  a: `${role}.right`,
  b: `${role}.left`,
  region: semanticRegion(role),
  support: "bilateral",
  use: ["outline", "nose", "mouth"].includes(semanticRegion(role)) ? "target" : "guardrail",
}));

const sideTemplates = [
  ["eye_aperture", "Eye aperture", "eye.upperApex", "eye.lowerApex", "eye"],
  ["eye_outer_aperture", "Outer eye aperture", "eye.upperOuter", "eye.lowerOuter", "eye"],
  ["eye_inner_aperture", "Inner eye aperture", "eye.upperInner", "eye.lowerInner", "eye"],
  ["brow_eye_inner", "Inner brow-eye spacing", "brow.innerBottom", "eye.upperInner", "brow"],
  ["brow_eye_mid", "Mid brow-eye spacing", "brow.midBottom", "eye.upperApex", "brow"],
  ["brow_eye_outer", "Outer brow-eye spacing", "brow.outerBottom", "eye.upperOuter", "brow"],
  ["brow_arch", "Brow arch height", "brow.archTop", "brow.midBottom", "brow"],
  ["nose_side_height", "Nasal side height", "nose.sideUpper", "nose.alarLower", "nose"],
  ["nose_side_lower", "Lower nasal side length", "nose.sideMid", "nose.sideLower", "nose"],
  ["alar_nostril", "Alar-to-nostril span", "nose.alarWidest", "nose.nostrilLateral", "nose"],
  ["upper_lip_side", "Upper lip side height", "mouth.upperOuterMid", "mouth.innerUpper", "mouth"],
  ["lower_lip_side", "Lower lip side height", "mouth.lowerOuterMid", "mouth.innerLower", "mouth"],
  ["corner_to_cupid", "Commissure-to-cupid span", "mouth.commissure", "mouth.cupidPeak", "mouth"],
  ["upper_jaw", "Upper jaw segment", "outline.zygionProxy", "outline.gonionProxy", "jaw"],
  ["lower_jaw", "Lower jaw segment", "outline.gonionProxy", "outline.chinLateral", "jaw"],
  ["malar_medial", "Malar-to-medial cheek span", "cheek.malarHigh", "cheek.medial", "cheek"],
] as const;

const sideSeeds: DistanceSeed[] = sideTemplates.flatMap(([id, label, a, b, region]) =>
  (["right", "left"] as const).map((side) => ({
    id: `distance.side.${id}.${side}`,
    label: `${side} ${label.toLowerCase()}`,
    a: `${a}.${side}`,
    b: `${b}.${side}`,
    region,
    support: "visible-side" as const,
    use: ["jaw", "nose", "mouth", "brow"].includes(region) ? "target" as const : "guardrail" as const,
  })),
);

const profileSeeds: DistanceSeed[] = [
  ["forehead_tip", "Forehead-to-nose-tip profile span", "forehead.glabellaProxy", "nose.pronasaleProxy"],
  ["nasion_tip", "Nasion-to-tip profile span", "nose.nasionProxy", "nose.pronasaleProxy"],
  ["tip_base", "Nose tip-to-base profile span", "nose.pronasaleProxy", "nose.subnasaleProxy"],
  ["base_upper_lip", "Nasal base-to-upper-lip profile span", "nose.subnasaleProxy", "mouth.labraleSuperius"],
  ["upper_lower_lip", "Upper-to-lower lip profile span", "mouth.labraleSuperius", "mouth.labraleInferius"],
  ["lower_lip_chin", "Lower-lip-to-chin profile span", "mouth.labraleInferius", "chin.menton"],
  ["nasion_chin", "Nasion-to-chin profile span", "nose.nasionProxy", "chin.menton"],
  ["tip_chin", "Tip-to-chin profile span", "nose.pronasaleProxy", "chin.menton"],
  ["glabella_chin", "Glabella-to-chin profile span", "forehead.glabellaProxy", "chin.menton"],
  ["bridge_chin", "Bridge-to-chin profile span", "nose.bridgeUpper", "chin.menton"],
].map(([id, label, a, b]) => ({ id: `distance.profile.${id}`, label, a, b, region: "profile", support: "profile", use: "guardrail" }));

const distanceSeeds = [...midlineDistanceSeeds, ...widthSeeds, ...sideSeeds, ...profileSeeds];
if (distanceSeeds.length !== 78) throw new Error(`Expected 78 distance measurements, got ${distanceSeeds.length}.`);

const poseListForSupport = (support: PoseSupport): PoseClass[] => {
  if (support === "frontal") return ["frontal"];
  if (support === "bilateral") return ["frontal", "three-quarter-left", "three-quarter-right"];
  if (support === "profile") return ["three-quarter-left", "three-quarter-right", "profile-left", "profile-right"];
  return [...ALL_SUPPORTED_POSES];
};

const distanceDefinitions: MeasurementDefinition[] = distanceSeeds.map((seed) => ({
  id: seed.id,
  label: seed.label,
  kind: "distance",
  region: seed.region,
  inputs: [seed.a, seed.b],
  poseSupport: seed.support,
  validPoses: poseListForSupport(seed.support),
  plannerUse: seed.use ?? "guardrail",
  consumerRules: seed.use === "target" ? ["harmony.proportion", "dimorphism.angularity"] : ["planner.geometry-guardrail"],
  unit: "pixels",
  operation: "distance",
}));

const FACE_WIDTH_ID = "distance.width.outline.zygionProxy";
const FACE_HEIGHT_ID = "distance.midline.face_height";
const LOWER_FACE_ID = "distance.midline.lower_face";
const EYE_SPAN_ID = "distance.width.eye.outerCanthus";
const INTERCANTHAL_ID = "distance.width.eye.innerCanthus";
const NOSE_WIDTH_ID = "distance.width.nose.alarWidest";
const MOUTH_WIDTH_ID = "distance.width.mouth.commissure";
const JAW_WIDTH_ID = "distance.width.outline.gonionProxy";

const firstNormalizer = (seed: DistanceSeed) => {
  if (seed.id === FACE_WIDTH_ID) return FACE_HEIGHT_ID;
  if (seed.id === FACE_HEIGHT_ID) return FACE_WIDTH_ID;
  if (["frame", "profile"].includes(seed.region)) return FACE_WIDTH_ID;
  if (["eye", "brow"].includes(seed.region)) return EYE_SPAN_ID;
  if (seed.region === "nose") return FACE_WIDTH_ID;
  if (seed.region === "mouth") return FACE_WIDTH_ID;
  if (["jaw", "chin", "outline", "cheek"].includes(seed.region)) return FACE_WIDTH_ID;
  return FACE_HEIGHT_ID;
};

const secondNormalizer = (seed: DistanceSeed) => {
  if (["eye", "brow"].includes(seed.region)) return INTERCANTHAL_ID;
  if (seed.region === "nose") return INTERCANTHAL_ID;
  if (seed.region === "mouth") return MOUTH_WIDTH_ID === seed.id ? NOSE_WIDTH_ID : MOUTH_WIDTH_ID;
  if (["jaw", "chin", "outline", "cheek"].includes(seed.region)) return JAW_WIDTH_ID === seed.id ? LOWER_FACE_ID : JAW_WIDTH_ID;
  return LOWER_FACE_ID === seed.id ? EYE_SPAN_ID : LOWER_FACE_ID;
};

const thirdNormalizer = (seed: DistanceSeed) => {
  if (["eye", "brow"].includes(seed.region)) return FACE_HEIGHT_ID;
  if (seed.region === "nose") return NOSE_WIDTH_ID === seed.id ? FACE_HEIGHT_ID : NOSE_WIDTH_ID;
  if (seed.region === "mouth") return NOSE_WIDTH_ID;
  if (["jaw", "chin", "outline", "cheek"].includes(seed.region)) return LOWER_FACE_ID;
  return EYE_SPAN_ID;
};

const ratioDefinition = (
  id: string,
  label: string,
  numerator: string,
  denominator: string,
  region: string,
  support: PoseSupport,
  use: PlannerUse,
): MeasurementDefinition => ({
  id,
  label,
  kind: "ratio",
  region,
  inputs: [numerator, denominator],
  poseSupport: support,
  validPoses: poseListForSupport(support),
  plannerUse: use,
  consumerRules: use === "target" ? ["harmony.reference-band", "planner.interaction-check"] : ["planner.ratio-guardrail"],
  unit: "ratio",
  operation: "ratio",
});

const primaryRatios = distanceSeeds.map((seed) => ratioDefinition(
  `ratio.primary.${seed.id.replace(/^distance\./, "")}`,
  `${seed.label} normalized`,
  seed.id,
  firstNormalizer(seed),
  seed.region,
  seed.support,
  seed.use ?? "guardrail",
));

const crossRatios = distanceSeeds.slice(0, 72).map((seed) => ratioDefinition(
  `ratio.cross.${seed.id.replace(/^distance\./, "")}`,
  `${seed.label} cross-feature ratio`,
  seed.id,
  secondNormalizer(seed),
  seed.region,
  seed.support,
  seed.use ?? "guardrail",
));

const partitionRatios = distanceSeeds.slice(0, 36).map((seed) => ratioDefinition(
  `ratio.partition.${seed.id.replace(/^distance\./, "")}`,
  `${seed.label} partition ratio`,
  seed.id,
  thirdNormalizer(seed),
  seed.region,
  seed.support,
  seed.use ?? "guardrail",
));

const sideRatios = sideTemplates.slice(0, 12).flatMap(([id, label, , , region]) => {
  const right = `distance.side.${id}.right`;
  const left = `distance.side.${id}.left`;
  return [
    ratioDefinition(`ratio.side.${id}.right_to_left`, `${label} right-to-left ratio`, right, left, region, "frontal", "guardrail"),
    ratioDefinition(`ratio.side.${id}.left_to_right`, `${label} left-to-right ratio`, left, right, region, "frontal", "guardrail"),
  ];
});

const ratioDefinitions = [...primaryRatios, ...crossRatios, ...partitionRatios, ...sideRatios];
if (ratioDefinitions.length !== 210) throw new Error(`Expected 210 ratio measurements, got ${ratioDefinitions.length}.`);

type AngleSeed = {
  id: string;
  label: string;
  a: string;
  vertex: string;
  b: string;
  region: string;
  support: PoseSupport;
};

const angleTemplates = [
  ["jaw_angle", "Jaw angle", "outline.zygionProxy", "outline.gonionProxy", "outline.chinLateral", "jaw", "visible-side"],
  ["chin_transition", "Chin transition angle", "outline.gonionProxy", "outline.chinLateral", "chin.menton", "chin", "visible-side"],
  ["temple_contour", "Temple contour angle", "outline.frontotemporaleProxy", "outline.templeLow", "outline.zygionProxy", "outline", "visible-side"],
  ["cheek_contour", "Cheek contour angle", "outline.templeLow", "outline.zygionProxy", "outline.gonionProxy", "cheek", "visible-side"],
  ["brow_arch", "Brow arch angle", "brow.innerBottom", "brow.archBottom", "brow.outerBottom", "brow", "visible-side"],
  ["brow_upper_arch", "Upper brow arch angle", "brow.innerTop", "brow.archTop", "brow.outerBottom", "brow", "visible-side"],
  ["eye_upper", "Upper eyelid angle", "eye.upperOuter", "eye.upperApex", "eye.upperInner", "eye", "visible-side"],
  ["eye_lower", "Lower eyelid angle", "eye.lowerOuter", "eye.lowerApex", "eye.lowerInner", "eye", "visible-side"],
  ["nose_side", "Nasal side angle", "nose.sideUpper", "nose.sideMid", "nose.sideLower", "nose", "visible-side"],
  ["nose_base", "Nasal base angle", "nose.sideLower", "nose.nostrilLateral", "nose.alarWidest", "nose", "visible-side"],
  ["upper_lip", "Upper lip contour angle", "mouth.commissure", "mouth.upperOuterMid", "mouth.cupidPeak", "mouth", "visible-side"],
  ["lower_lip", "Lower lip contour angle", "mouth.commissure", "mouth.lowerOuterMid", "mouth.lowerOuterMedial", "mouth", "visible-side"],
  ["inner_lip", "Inner lip contour angle", "mouth.innerCommissure", "mouth.innerUpper", "mouth.innerLower", "mouth", "visible-side"],
  ["jaw_lower", "Lower jaw contour angle", "outline.gonionProxy", "outline.jawMid", "outline.preChin", "jaw", "visible-side"],
  ["prechin", "Pre-chin contour angle", "outline.jawMid", "outline.preChin", "chin.menton", "chin", "visible-side"],
  ["malar", "Malar triangle angle", "cheek.malarHigh", "cheek.malarLateral", "cheek.medial", "cheek", "visible-side"],
  ["orbital_brow", "Outer orbital-brow angle", "eye.outerCanthus", "brow.outerBottom", "brow.archBottom", "brow", "visible-side"],
  ["profile_convexity", "Visible profile convexity proxy", "outline.frontotemporaleProxy", "nose.alarWidest", "outline.chinLateral", "profile", "profile"],
] as const;

const angleSeeds: AngleSeed[] = angleTemplates.flatMap(([id, label, a, vertex, b, region, support]) =>
  (["right", "left"] as const).map((side) => ({
    id: `angle.${id}.${side}`,
    label: `${side} ${label.toLowerCase()}`,
    a: a.includes("chin.menton") ? a : `${a}.${side}`,
    vertex: vertex.includes("chin.menton") ? vertex : `${vertex}.${side}`,
    b: b.includes("chin.menton") ? b : `${b}.${side}`,
    region,
    support: support as PoseSupport,
  })),
);

const angleDefinitions: MeasurementDefinition[] = angleSeeds.map((seed) => ({
  id: seed.id,
  label: seed.label,
  kind: "angle",
  region: seed.region,
  inputs: [seed.a, seed.vertex, seed.b],
  poseSupport: seed.support,
  validPoses: poseListForSupport(seed.support),
  plannerUse: ["jaw", "chin", "nose", "mouth", "brow"].includes(seed.region) ? "guardrail" : "context",
  consumerRules: ["planner.angle-guardrail"],
  unit: "degrees",
  operation: "angle",
}));
if (angleDefinitions.length !== 36) throw new Error(`Expected 36 angle measurements, got ${angleDefinitions.length}.`);

type ShapeSeed = {
  id: string;
  label: string;
  operation: "polyline" | "curvature" | "area";
  inputs: string[];
  region: string;
  support: PoseSupport;
};

const sided = (roles: string[], side: "right" | "left") => roles.map((role) => `${role}.${side}`);
const shapeSeeds: ShapeSeed[] = [
  { id: "shape.path.outline.right", label: "Right outline path", operation: "polyline", inputs: sided(["outline.frontotemporaleProxy", "outline.templeLow", "outline.zygionProxy", "outline.gonionProxy", "outline.jawMid", "outline.chinLateral"], "right"), region: "outline", support: "visible-side" },
  { id: "shape.path.outline.left", label: "Left outline path", operation: "polyline", inputs: sided(["outline.frontotemporaleProxy", "outline.templeLow", "outline.zygionProxy", "outline.gonionProxy", "outline.jawMid", "outline.chinLateral"], "left"), region: "outline", support: "visible-side" },
  { id: "shape.path.brow.right", label: "Right brow path", operation: "polyline", inputs: sided(["brow.outerBottom", "brow.archBottom", "brow.midBottom", "brow.innerBottom"], "right"), region: "brow", support: "visible-side" },
  { id: "shape.path.brow.left", label: "Left brow path", operation: "polyline", inputs: sided(["brow.outerBottom", "brow.archBottom", "brow.midBottom", "brow.innerBottom"], "left"), region: "brow", support: "visible-side" },
  { id: "shape.path.eye.right", label: "Right upper eye path", operation: "polyline", inputs: sided(["eye.outerCanthus", "eye.upperOuter", "eye.upperApex", "eye.upperInner", "eye.innerCanthus"], "right"), region: "eye", support: "visible-side" },
  { id: "shape.path.eye.left", label: "Left upper eye path", operation: "polyline", inputs: sided(["eye.outerCanthus", "eye.upperOuter", "eye.upperApex", "eye.upperInner", "eye.innerCanthus"], "left"), region: "eye", support: "visible-side" },
  { id: "shape.path.nose.right", label: "Right nasal side path", operation: "polyline", inputs: sided(["nose.sideUpper", "nose.sideMid", "nose.sideLower", "nose.alarLower", "nose.alarWidest"], "right"), region: "nose", support: "visible-side" },
  { id: "shape.path.nose.left", label: "Left nasal side path", operation: "polyline", inputs: sided(["nose.sideUpper", "nose.sideMid", "nose.sideLower", "nose.alarLower", "nose.alarWidest"], "left"), region: "nose", support: "visible-side" },
  { id: "shape.curvature.jaw.right", label: "Right jaw curvature", operation: "curvature", inputs: sided(["outline.zygionProxy", "outline.gonionProxy", "outline.jawMid", "outline.preChin", "outline.chinLateral"], "right"), region: "jaw", support: "visible-side" },
  { id: "shape.curvature.jaw.left", label: "Left jaw curvature", operation: "curvature", inputs: sided(["outline.zygionProxy", "outline.gonionProxy", "outline.jawMid", "outline.preChin", "outline.chinLateral"], "left"), region: "jaw", support: "visible-side" },
  { id: "shape.curvature.brow.right", label: "Right brow curvature", operation: "curvature", inputs: sided(["brow.outerBottom", "brow.archBottom", "brow.midBottom", "brow.innerBottom"], "right"), region: "brow", support: "visible-side" },
  { id: "shape.curvature.brow.left", label: "Left brow curvature", operation: "curvature", inputs: sided(["brow.outerBottom", "brow.archBottom", "brow.midBottom", "brow.innerBottom"], "left"), region: "brow", support: "visible-side" },
  { id: "shape.curvature.lip.right", label: "Right upper-lip curvature", operation: "curvature", inputs: sided(["mouth.commissure", "mouth.upperOuterLateral", "mouth.upperOuterMid", "mouth.cupidPeak"], "right"), region: "mouth", support: "visible-side" },
  { id: "shape.curvature.lip.left", label: "Left upper-lip curvature", operation: "curvature", inputs: sided(["mouth.commissure", "mouth.upperOuterLateral", "mouth.upperOuterMid", "mouth.cupidPeak"], "left"), region: "mouth", support: "visible-side" },
  { id: "shape.curvature.nose.right", label: "Right nose curvature", operation: "curvature", inputs: sided(["nose.sideUpper", "nose.sideMid", "nose.sideLower", "nose.alarWidest"], "right"), region: "nose", support: "visible-side" },
  { id: "shape.curvature.nose.left", label: "Left nose curvature", operation: "curvature", inputs: sided(["nose.sideUpper", "nose.sideMid", "nose.sideLower", "nose.alarWidest"], "left"), region: "nose", support: "visible-side" },
  { id: "shape.area.eye.right", label: "Right eye aperture area", operation: "area", inputs: sided(["eye.outerCanthus", "eye.upperApex", "eye.innerCanthus", "eye.lowerApex"], "right"), region: "eye", support: "visible-side" },
  { id: "shape.area.eye.left", label: "Left eye aperture area", operation: "area", inputs: sided(["eye.outerCanthus", "eye.upperApex", "eye.innerCanthus", "eye.lowerApex"], "left"), region: "eye", support: "visible-side" },
  { id: "shape.area.mouth", label: "Central mouth aperture area", operation: "area", inputs: ["mouth.innerCommissure.right", "mouth.stomionUpper", "mouth.innerCommissure.left", "mouth.stomionLower"], region: "mouth", support: "frontal" },
  { id: "shape.area.nose", label: "Nasal base proxy area", operation: "area", inputs: ["nose.alarWidest.right", "nose.pronasaleProxy", "nose.alarWidest.left", "nose.subnasaleProxy"], region: "nose", support: "frontal" },
];

const shapeDefinitions: MeasurementDefinition[] = shapeSeeds.map((seed) => ({
  id: seed.id,
  label: seed.label,
  kind: "shape",
  region: seed.region,
  inputs: seed.inputs,
  poseSupport: seed.support,
  validPoses: poseListForSupport(seed.support),
  plannerUse: "guardrail",
  consumerRules: ["planner.contour-guardrail"],
  unit: seed.operation === "area" ? "normalized" : seed.operation === "curvature" ? "degrees" : "pixels",
  operation: seed.operation,
}));
if (shapeDefinitions.length !== 20) throw new Error(`Expected 20 shape measurements, got ${shapeDefinitions.length}.`);

const symmetryRoles = [
  "outline.frontotemporaleProxy", "outline.templeLow", "outline.zygionProxy", "outline.cheek",
  "outline.gonionProxy", "outline.jawAngleLower", "outline.jawMid", "outline.chinLateral",
  "brow.innerBottom", "brow.midBottom", "brow.archBottom", "brow.outerBottom",
  "eye.outerCanthus", "eye.innerCanthus", "eye.upperApex", "eye.lowerApex",
  "nose.alarWidest", "nose.alarBase", "mouth.commissure", "mouth.cupidPeak",
];

const symmetryDefinitions: MeasurementDefinition[] = symmetryRoles.flatMap((role) =>
  (["axial", "vertical"] as const).map((operation) => ({
    id: `symmetry.${operation}.${role}`,
    label: `${role.replaceAll(".", " ")} ${operation} residual`,
    kind: "symmetry" as const,
    region: semanticRegion(role),
    inputs: [`${role}.right`, `${role}.left`],
    poseSupport: "frontal" as const,
    validPoses: ["frontal" as const],
    plannerUse: "target" as const,
    consumerRules: ["symmetry.natural-deadband"],
    unit: "ratio" as const,
    operation,
  })),
);
if (symmetryDefinitions.length !== 40) throw new Error(`Expected 40 symmetry measurements, got ${symmetryDefinitions.length}.`);

const contextIds = [
  "pose.yaw", "pose.pitch", "pose.roll", "pose.matrix_available", "pose.confidence", "pose.frontal_weight",
  "expression.jaw_open", "expression.mouth_pucker", "expression.mouth_funnel", "expression.smile", "expression.blink",
  "expression.squint", "expression.brow_activity", "expression.cheek_activity", "expression.confidence", "expression.neutral",
  "capture.quality", "capture.temporal_stability", "capture.resolution", "capture.landmark_coverage", "capture.face_scale",
  "capture.exposure", "capture.local_mesh_consistency", "capture.overall_confidence",
];

const contextDefinitions: MeasurementDefinition[] = contextIds.map((id) => ({
  id,
  label: id.replaceAll(".", " "),
  kind: "context",
  region: id.split(".")[0],
  inputs: [],
  poseSupport: "all",
  validPoses: [...ALL_SUPPORTED_POSES, "unsupported"],
  plannerUse: id.includes("confidence") || id.includes("quality") ? "confidence" : "context",
  consumerRules: ["planner.confidence-gate"],
  unit: "normalized",
}));
if (contextDefinitions.length !== 24) throw new Error(`Expected 24 context measurements, got ${contextDefinitions.length}.`);

export const MEASUREMENT_CATALOG: MeasurementDefinition[] = [
  ...distanceDefinitions,
  ...ratioDefinitions,
  ...angleDefinitions,
  ...shapeDefinitions,
  ...symmetryDefinitions,
  ...contextDefinitions,
];

export const MEASUREMENT_COUNTS = {
  distance: 78,
  ratio: 210,
  angle: 36,
  shape: 20,
  symmetry: 40,
  context: 24,
  total: 408,
} as const;

if (MEASUREMENT_CATALOG.length !== MEASUREMENT_COUNTS.total) {
  throw new Error(`Expected 408 measurements, got ${MEASUREMENT_CATALOG.length}.`);
}
if (new Set(MEASUREMENT_CATALOG.map((definition) => definition.id)).size !== MEASUREMENT_CATALOG.length) {
  throw new Error("Measurement IDs must be unique.");
}

type SemanticPoint = Point & { reliability: number };

function averagePoints(points: Point[]): Point | undefined {
  if (!points.length) return undefined;
  return {
    x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
    y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
    z: points.reduce((sum, point) => sum + (point.z ?? 0), 0) / points.length,
  };
}

function resolveSemanticPoints(observation: FaceObservation, width: number, height: number) {
  const map = new Map<string, SemanticPoint>();
  const depthScale = Math.sqrt(width * height);
  for (const definition of SEMANTIC_LANDMARKS) {
    let point = observation.landmarks[definition.meshIndex];
    if (!point && definition.fallbackIndices) {
      point = averagePoints(
        definition.fallbackIndices
          .map((index) => observation.landmarks[index])
          .filter((candidate): candidate is Point => Boolean(candidate)),
      );
    }
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
    map.set(definition.id, {
      x: point.x * width,
      y: point.y * height,
      z: (point.z ?? 0) * depthScale,
      reliability: definition.reliability,
    });
  }
  return map;
}

const radiansToDegrees = (value: number) => value * (180 / Math.PI);

function matrixPose(matrix: FaceObservation["transformationMatrix"]) {
  if (!matrix || matrix.data.length < 16 || matrix.rows < 4 || matrix.columns < 4) return null;
  const m = matrix.data;
  const column = (index: number) => [m[index * 4], m[index * 4 + 1], m[index * 4 + 2]];
  const [c0, c1, c2] = [column(0), column(1), column(2)];
  const scale = ([...c0, ...c1, ...c2].reduce((sum, value) => sum + value * value, 0) / 3) ** 0.5;
  if (!Number.isFinite(scale) || scale < 0.001) return null;
  const r00 = c0[0] / scale;
  const r10 = c0[1] / scale;
  const r20 = c0[2] / scale;
  const r21 = c1[2] / scale;
  const r22 = c2[2] / scale;
  const yaw = radiansToDegrees(Math.atan2(-r20, Math.hypot(r00, r10)));
  const pitch = radiansToDegrees(Math.atan2(r21, r22));
  const roll = radiansToDegrees(Math.atan2(r10, r00));
  if (![yaw, pitch, roll].every(Number.isFinite) || Math.abs(yaw) > 100 || Math.abs(pitch) > 100 || Math.abs(roll) > 100) return null;
  return { yaw, pitch, roll };
}

export function estimatePose(observation: FaceObservation, width = 1, height = 1): PoseEstimate {
  const points = observation.landmarks;
  const point = (index: number) => ({ x: (points[index]?.x ?? 0.5) * width, y: (points[index]?.y ?? 0.5) * height });
  const eyeRight = point(33);
  const eyeLeft = point(263);
  const roll2d = radiansToDegrees(Math.atan2(eyeLeft.y - eyeRight.y, eyeLeft.x - eyeRight.x));
  const nose = point(1);
  const cheekRight = point(234);
  const cheekLeft = point(454);
  const rightSpan = Math.hypot(nose.x - cheekRight.x, nose.y - cheekRight.y);
  const leftSpan = Math.hypot(cheekLeft.x - nose.x, cheekLeft.y - nose.y);
  const yawSignal = clamp((leftSpan - rightSpan) / Math.max(leftSpan + rightSpan, 1), -1, 1);
  const yawFromProjection = yawSignal * 72;
  const transformed = matrixPose(observation.transformationMatrix);
  const matrixWeight = transformed ? 0.72 : 0;
  const yawSign = Math.sign((transformed?.yaw ?? 0) || yawFromProjection || 1);
  const yawMagnitude = transformed
    ? Math.max(Math.abs(transformed.yaw) * matrixWeight + Math.abs(yawFromProjection) * (1 - matrixWeight), Math.abs(yawFromProjection) * 0.78)
    : Math.abs(yawFromProjection);
  const yawDeg = yawSign * yawMagnitude;
  const pitchDeg = transformed?.pitch ?? 0;
  const rollDeg = Number.isFinite(roll2d) ? roll2d : transformed?.roll ?? 0;
  const matrixAgreement = transformed
    ? 1 - clamp(Math.abs(Math.abs(transformed.yaw) - Math.abs(yawFromProjection)) / 55)
    : 0.68;
  let confidence = clamp((transformed ? 0.86 : 0.66) * (0.72 + matrixAgreement * 0.28));
  const yaw = Math.abs(yawDeg);
  let poseClass: PoseClass;
  if (yaw <= 16) poseClass = "frontal";
  else if (yaw <= 46) poseClass = yawDeg >= 0 ? "three-quarter-right" : "three-quarter-left";
  else if (yaw <= 78) poseClass = yawDeg >= 0 ? "profile-right" : "profile-left";
  else poseClass = "unsupported";
  if (Math.abs(pitchDeg) > 34 || Math.abs(rollDeg) > 24) poseClass = "unsupported";
  if (poseClass === "unsupported") confidence *= 0.45;
  else if (poseClass.includes("profile")) confidence *= 0.86;
  else if (poseClass.includes("three-quarter")) confidence *= 0.93;
  const visibleSide = poseClass === "frontal" ? "both" : yawDeg >= 0 ? "right" : "left";
  const label = poseClass === "frontal"
    ? "Frontal"
    : poseClass === "three-quarter-left"
      ? "Three-quarter left"
      : poseClass === "three-quarter-right"
        ? "Three-quarter right"
        : poseClass === "profile-left"
          ? "Left profile"
          : poseClass === "profile-right"
            ? "Right profile"
            : "Unsupported pose";
  return { class: poseClass, label, yawDeg, pitchDeg, rollDeg, confidence: clamp(confidence), visibleSide };
}

function blendshapeScore(observation: FaceObservation, ...names: string[]) {
  return names.reduce((maximum, name) => Math.max(maximum, observation.blendshapes[name] ?? 0), 0);
}

function readExpression(observation: FaceObservation) {
  const jawOpen = blendshapeScore(observation, "jawOpen");
  const rawPucker = blendshapeScore(observation, "mouthPucker", "mouthFunnel");
  const distance = (a: number, b: number) => Math.hypot(
    observation.landmarks[a].x - observation.landmarks[b].x,
    observation.landmarks[a].y - observation.landmarks[b].y,
  );
  const mouthWidth = distance(61, 291);
  const faceWidth = Math.max(distance(234, 454), 0.001);
  const noseWidth = Math.max(distance(98, 327), 0.001);
  const mouthToFace = mouthWidth / faceWidth;
  const mouthToNose = mouthWidth / noseWidth;
  // MediaPipe can report a strong mouthPucker coefficient for naturally full,
  // closed lips. Require supporting 2D narrowing before that coefficient can
  // fully lock the lips and chin; retain a 40% floor so a true pucker remains
  // conservative even when perspective makes the width cue ambiguous.
  const puckerGeometry = Math.max(
    clamp((0.3 - mouthToFace) / 0.1),
    clamp((1.3 - mouthToNose) / 0.38),
  );
  const pucker = rawPucker * (0.4 + puckerGeometry * 0.6);
  const smile = blendshapeScore(observation, "mouthSmileLeft", "mouthSmileRight");
  const blink = blendshapeScore(observation, "eyeBlinkLeft", "eyeBlinkRight");
  const squint = blendshapeScore(observation, "eyeSquintLeft", "eyeSquintRight");
  const brow = blendshapeScore(observation, "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight");
  const cheek = blendshapeScore(observation, "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight");
  const structuralActivity = Math.max(jawOpen, pucker, smile * 0.72, blink * 0.7, squint * 0.68, brow * 0.65, cheek * 0.7);
  const confidence = clamp(1 - structuralActivity * 0.88);
  const blockedRegions: string[] = [];
  if (jawOpen > 0.22 || pucker > 0.32 || smile > 0.62) blockedRegions.push("Lips", "Chin");
  if (brow > 0.58) blockedRegions.push("Brows");
  if (blink > 0.65 || squint > 0.6) blockedRegions.push("Symmetry");
  if (cheek > 0.62) blockedRegions.push("Face shape");
  const label = confidence < 0.52 ? "Expressive — edits limited" : smile > 0.22 ? "Mild smile" : "Neutral";
  return {
    label,
    confidence,
    blockedRegions: [...new Set(blockedRegions)],
    values: { jawOpen, pucker, funnel: blendshapeScore(observation, "mouthFunnel"), smile, blink, squint, brow, cheek },
  };
}

function supportConfidence(support: PoseSupport, pose: PoseEstimate) {
  if (pose.class === "unsupported") return 0;
  const frontal = pose.class === "frontal";
  const threeQuarter = pose.class.includes("three-quarter");
  if (support === "frontal") return frontal ? 1 : 0;
  if (support === "bilateral") return frontal ? 1 : threeQuarter ? 0.48 : 0;
  if (support === "profile") return frontal ? 0.08 : threeQuarter ? 0.48 : 0.92;
  if (support === "visible-side") return frontal ? 0.94 : threeQuarter ? 0.86 : 0.76;
  return frontal ? 1 : threeQuarter ? 0.82 : 0.62;
}

function expressionConfidenceForRegion(region: string, expression: ReturnType<typeof readExpression>) {
  const lower = region.toLowerCase();
  if (lower.includes("mouth") || lower.includes("lip")) {
    return clamp(1 - Math.max(expression.values.jawOpen, expression.values.pucker, expression.values.smile * 0.75));
  }
  if (lower.includes("brow") || lower.includes("eye")) {
    return clamp(1 - Math.max(expression.values.brow, expression.values.blink, expression.values.squint) * 0.8);
  }
  if (lower.includes("cheek") || lower.includes("outline")) return clamp(1 - expression.values.cheek * 0.7);
  return expression.confidence;
}

function buildRegionEditability(
  pose: PoseEstimate,
  expression: ReturnType<typeof readExpression>,
  quality: number,
): RegionEditability[] {
  const isFront = pose.class === "frontal";
  const isThreeQuarter = pose.class.includes("three-quarter");
  const bases: Record<FaceRegion, number> = isFront
    ? { Jaw: 0.88, Chin: 0.86, Nose: 0.84, Lips: 0.78, Brows: 0.8, "Face shape": 0.78, Symmetry: 0.86 }
    : isThreeQuarter
      ? { Jaw: 0.72, Chin: 0.78, Nose: 0.64, Lips: 0.63, Brows: 0.62, "Face shape": 0.6, Symmetry: 0.16 }
      : pose.class.includes("profile")
        ? { Jaw: 0.64, Chin: 0.78, Nose: 0.68, Lips: 0.62, Brows: 0.38, "Face shape": 0.54, Symmetry: 0 }
        : { Jaw: 0, Chin: 0, Nose: 0, Lips: 0, Brows: 0, "Face shape": 0, Symmetry: 0 };
  const displacementCeilings: Record<FaceRegion, number> = {
    Jaw: 0.045,
    Chin: 0.035,
    Nose: 0.022,
    Lips: 0.022,
    Brows: 0.016,
    "Face shape": 0.035,
    Symmetry: 0.015,
  };
  return (Object.entries(bases) as Array<[FaceRegion, number]>).map(([region, base]) => {
    const expressionFactor = expression.blockedRegions.includes(region) ? 0.32 : region === "Lips" ? expressionConfidenceForRegion("mouth", expression) : region === "Brows" ? expressionConfidenceForRegion("brow", expression) : expression.confidence;
    const score = clamp(base * (0.55 + quality * 0.45) * (0.62 + expressionFactor * 0.38) * (0.72 + pose.confidence * 0.28));
    const reasons: string[] = [];
    if (pose.class === "unsupported") reasons.push("Pose falls outside V2's tested range");
    else if (region === "Symmetry" && !isFront) reasons.push("Perspective can imitate asymmetry at this angle");
    else if (pose.class.includes("profile") && ["Brows", "Face shape", "Jaw"].includes(region)) reasons.push("Only the visible silhouette is editable in profile");
    else if (isThreeQuarter && ["Nose", "Lips", "Face shape"].includes(region)) reasons.push("Far-side geometry is downweighted");
    if (expression.blockedRegions.includes(region)) reasons.push("Expression makes this region unreliable");
    if (quality < 0.72) reasons.push("Capture quality reduces the displacement budget");
    if (!reasons.length) reasons.push("Pose, expression, and pixel support are adequate");
    return {
      region,
      score,
      editable: score >= 0.55,
      reasons,
      maxDisplacement: displacementCeilings[region] * (0.55 + score * 0.45),
    };
  });
}

function angleAt(a: Point, vertex: Point, b: Point) {
  const ax = a.x - vertex.x;
  const ay = a.y - vertex.y;
  const bx = b.x - vertex.x;
  const by = b.y - vertex.y;
  const denominator = Math.max(Math.hypot(ax, ay) * Math.hypot(bx, by), 1e-6);
  return radiansToDegrees(Math.acos(clamp((ax * bx + ay * by) / denominator, -1, 1)));
}

function polygonArea(points: Point[]) {
  let area = 0;
  for (let index = 0; index < points.length; index += 1) {
    const next = points[(index + 1) % points.length];
    area += points[index].x * next.y - next.x * points[index].y;
  }
  return Math.abs(area) / 2;
}

function coordinateFrame(points: Point[], width: number, height: number) {
  const pixel = (index: number) => ({ x: (points[index]?.x ?? 0.5) * width, y: (points[index]?.y ?? 0.5) * height });
  const rightEye = pixel(33);
  const leftEye = pixel(263);
  const eyeSpan = Math.max(Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y), 1e-6);
  const ux = (leftEye.x - rightEye.x) / eyeSpan;
  const uy = (leftEye.y - rightEye.y) / eyeSpan;
  const vx = -uy;
  const vy = ux;
  const forehead = pixel(10);
  const chin = pixel(152);
  const origin = { x: (forehead.x + chin.x) / 2, y: (forehead.y + chin.y) / 2 };
  return {
    local(point: Point) {
      const x = point.x - origin.x;
      const y = point.y - origin.y;
      return { u: x * ux + y * uy, v: x * vx + y * vy };
    },
  };
}

function finiteOrNull(value: number) {
  return Number.isFinite(value) ? value : null;
}

function measurementRegion(region: string): FaceRegion | null {
  if (["jaw", "outline"].includes(region)) return "Jaw";
  if (region === "chin") return "Chin";
  if (region === "nose" || region === "profile") return "Nose";
  if (region === "mouth") return "Lips";
  if (region === "brow" || region === "eye") return "Brows";
  if (region === "cheek" || region === "frame") return "Face shape";
  return null;
}

function hiddenSideFactor(definition: MeasurementDefinition, pose: PoseEstimate) {
  if (pose.visibleSide === "both") return 1;
  const mentionsLeft = definition.inputs.some((input) => input.endsWith(".left"));
  const mentionsRight = definition.inputs.some((input) => input.endsWith(".right"));
  if ((mentionsLeft && mentionsRight) || (!mentionsLeft && !mentionsRight)) return 1;
  const isVisible = pose.visibleSide === "left" ? mentionsLeft : mentionsRight;
  return isVisible ? 1 : pose.class.includes("profile") ? 0 : 0.48;
}

function evaluateMeasurements(
  observation: FaceObservation,
  semantic: Map<string, SemanticPoint>,
  pose: PoseEstimate,
  expression: ReturnType<typeof readExpression>,
  editability: RegionEditability[],
  width: number,
  height: number,
  quality: {
    quality: number;
    temporal: number;
    resolution: number;
    exposure: number;
    coverage: number;
    faceScale: number;
    meshConsistency: number;
    overall: number;
  },
) {
  const values = new Map<string, MeasurementValue>();
  const frame = coordinateFrame(observation.landmarks, width, height);
  const faceWidth = Math.max(
    Math.hypot((observation.landmarks[454]?.x ?? 0.7) * width - (observation.landmarks[234]?.x ?? 0.3) * width, (observation.landmarks[454]?.y ?? 0.5) * height - (observation.landmarks[234]?.y ?? 0.5) * height),
    1,
  );
  const faceHeight = Math.max(
    Math.hypot((observation.landmarks[152]?.x ?? 0.5) * width - (observation.landmarks[10]?.x ?? 0.5) * width, (observation.landmarks[152]?.y ?? 0.8) * height - (observation.landmarks[10]?.y ?? 0.2) * height),
    1,
  );
  const editabilityMap = new Map(editability.map((entry) => [entry.region, entry.score]));

  const addValue = (
    definition: MeasurementDefinition,
    value: number | null,
    inputConfidence: number,
    extraReasons: string[] = [],
    inheritedContext = false,
  ) => {
    const poseFactor = supportConfidence(definition.poseSupport, pose);
    const sideFactor = hiddenSideFactor(definition, pose);
    const expressionFactor = expressionConfidenceForRegion(definition.region, expression);
    const confidence = definition.kind === "context" || inheritedContext
      ? clamp(inputConfidence)
      : clamp(inputConfidence * poseFactor * sideFactor * (0.62 + quality.overall * 0.38) * (0.68 + expressionFactor * 0.32));
    const reasons = [...extraReasons];
    if (value === null) reasons.push("Required geometry was unavailable");
    if (poseFactor <= 0.01) reasons.push("Measurement is invalid for this pose");
    else if (poseFactor < 0.62) reasons.push("Perspective reduces measurement confidence");
    if (sideFactor === 0) reasons.push("Required landmarks are on the hidden side");
    if (expressionFactor < 0.5) reasons.push("Expression affects this measurement");
    const validity = value === null || confidence < 0.38 ? "invalid" : confidence < 0.62 ? "degraded" : "valid";
    const region = measurementRegion(definition.region);
    const regionScore = region ? editabilityMap.get(region) ?? confidence : confidence;
    const result: MeasurementValue = {
      id: definition.id,
      label: definition.label,
      kind: definition.kind,
      region: definition.region,
      value,
      confidence,
      editability: Math.min(confidence, regionScore),
      validity,
      reasons,
      plannerUse: definition.plannerUse,
    };
    values.set(definition.id, result);
  };

  for (const definition of distanceDefinitions) {
    const [a, b] = definition.inputs.map((input) => semantic.get(input));
    const reliability = a && b ? Math.min(a.reliability, b.reliability) : 0;
    addValue(definition, a && b ? finiteOrNull(Math.hypot(a.x - b.x, a.y - b.y)) : null, reliability * quality.temporal);
  }

  for (const definition of ratioDefinitions) {
    const numerator = values.get(definition.inputs[0]);
    const denominator = values.get(definition.inputs[1]);
    const denominatorValue = denominator?.value ?? 0;
    const usable = numerator?.value !== null && numerator?.value !== undefined && denominatorValue > 1e-6;
    addValue(
      definition,
      usable ? finiteOrNull((numerator?.value ?? 0) / denominatorValue) : null,
      Math.min(numerator?.confidence ?? 0, denominator?.confidence ?? 0),
      denominatorValue <= 1e-6 ? ["Normalizer was unstable"] : [],
      true,
    );
  }

  for (const definition of angleDefinitions) {
    const [a, vertex, b] = definition.inputs.map((input) => semantic.get(input));
    const reliability = a && vertex && b ? Math.min(a.reliability, vertex.reliability, b.reliability) : 0;
    addValue(definition, a && vertex && b ? finiteOrNull(angleAt(a, vertex, b)) : null, reliability * quality.temporal);
  }

  for (const definition of shapeDefinitions) {
    const points = definition.inputs.map((input) => semantic.get(input));
    const complete = points.every((point): point is SemanticPoint => Boolean(point));
    let value: number | null = null;
    if (complete && definition.operation === "polyline") {
      value = points.slice(1).reduce((sum, point, index) => sum + Math.hypot(point.x - points[index].x, point.y - points[index].y), 0);
    } else if (complete && definition.operation === "curvature") {
      const turns = points.slice(1, -1).map((point, index) => 180 - angleAt(points[index], point, points[index + 2]));
      value = turns.reduce((sum, turn) => sum + Math.abs(turn), 0) / Math.max(1, turns.length);
    } else if (complete && definition.operation === "area") {
      value = polygonArea(points) / Math.max(faceWidth * faceHeight, 1);
    }
    const reliability = complete ? Math.min(...points.map((point) => point.reliability)) : 0;
    addValue(definition, finiteOrNull(value ?? Number.NaN), reliability * quality.temporal);
  }

  for (const definition of symmetryDefinitions) {
    const [right, left] = definition.inputs.map((input) => semantic.get(input));
    let value: number | null = null;
    if (right && left) {
      const r = frame.local(right);
      const l = frame.local(left);
      const center = ((semantic.get("forehead.center") ? frame.local(semantic.get("forehead.center") as Point).u : 0) + (semantic.get("chin.menton") ? frame.local(semantic.get("chin.menton") as Point).u : 0)) / 2;
      value = definition.operation === "axial"
        ? Math.abs(Math.abs(r.u - center) - Math.abs(l.u - center)) / faceWidth
        : Math.abs(r.v - l.v) / faceHeight;
    }
    const reliability = right && left ? Math.min(right.reliability, left.reliability) : 0;
    addValue(definition, finiteOrNull(value ?? Number.NaN), reliability * quality.temporal);
  }

  const blend = expression.values;
  const contextValues: Record<string, number> = {
    "pose.yaw": Math.abs(pose.yawDeg) / 90,
    "pose.pitch": Math.abs(pose.pitchDeg) / 45,
    "pose.roll": Math.abs(pose.rollDeg) / 45,
    "pose.matrix_available": observation.transformationMatrix ? 1 : 0,
    "pose.confidence": pose.confidence,
    "pose.frontal_weight": supportConfidence("frontal", pose),
    "expression.jaw_open": blend.jawOpen,
    "expression.mouth_pucker": blend.pucker,
    "expression.mouth_funnel": blend.funnel,
    "expression.smile": blend.smile,
    "expression.blink": blend.blink,
    "expression.squint": blend.squint,
    "expression.brow_activity": blend.brow,
    "expression.cheek_activity": blend.cheek,
    "expression.confidence": expression.confidence,
    "expression.neutral": expression.label === "Neutral" ? 1 : 0,
    "capture.quality": quality.quality,
    "capture.temporal_stability": quality.temporal,
    "capture.resolution": quality.resolution,
    "capture.landmark_coverage": quality.coverage,
    "capture.face_scale": quality.faceScale,
    "capture.exposure": quality.exposure,
    "capture.local_mesh_consistency": quality.meshConsistency,
    "capture.overall_confidence": quality.overall,
  };
  for (const definition of contextDefinitions) {
    const contextConfidence = definition.id.startsWith("pose.")
      ? pose.confidence
      : definition.id.startsWith("expression.")
        ? expression.confidence
        : quality.overall;
    addValue(definition, finiteOrNull(contextValues[definition.id]), contextConfidence);
  }
  return MEASUREMENT_CATALOG.map((definition) => values.get(definition.id) as MeasurementValue);
}

function landmarkDistance(points: Point[], a: number, b: number, width: number, height: number) {
  const first = points[a];
  const second = points[b];
  if (!first || !second) return 0;
  return Math.hypot((first.x - second.x) * width, (first.y - second.y) * height);
}

export function analyzeFace(
  observation: FaceObservation,
  width = 1,
  height = 1,
  options: AnalysisOptions = {},
): FaceAnalysis {
  if (!observation.landmarks || observation.landmarks.length < 468) {
    throw new Error("A complete dense face mesh is required for pose-aware analysis.");
  }
  const pose = estimatePose(observation, width, height);
  const expression = readExpression(observation);
  const semantic = resolveSemanticPoints(observation, width, height);
  const qualityConfidence = clamp(options.qualityConfidence ?? 0.9);
  const temporalStability = clamp(options.temporalStability ?? 0.82);
  const resolutionSupport = clamp(options.resolutionSupport ?? 0.9);
  const exposureSupport = clamp(options.exposureSupport ?? 0.9);
  const coverage = clamp(semantic.size / SEMANTIC_LANDMARK_COUNT);
  const faceWidthNormalized = Math.abs((observation.landmarks[454]?.x ?? 0.7) - (observation.landmarks[234]?.x ?? 0.3));
  const faceScale = clamp(1 - Math.abs(faceWidthNormalized - 0.42) / 0.42);
  const meshConsistency = observation.landmarks.slice(0, 468).every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)) ? 0.94 : 0.4;
  const overallConfidence = clamp(Math.min(
    pose.confidence,
    qualityConfidence,
    0.55 + temporalStability * 0.45,
    0.55 + resolutionSupport * 0.45,
    0.55 + exposureSupport * 0.45,
    0.5 + coverage * 0.5,
  ) * (0.74 + expression.confidence * 0.26));
  const regionEditability = buildRegionEditability(pose, expression, qualityConfidence);
  const measurements = evaluateMeasurements(
    observation,
    semantic,
    pose,
    expression,
    regionEditability,
    width,
    height,
    {
      quality: qualityConfidence,
      temporal: temporalStability,
      resolution: resolutionSupport,
      exposure: exposureSupport,
      coverage,
      faceScale,
      meshConsistency,
      overall: overallConfidence,
    },
  );

  const faceWidth = Math.max(landmarkDistance(observation.landmarks, 234, 454, width, height), 0.001);
  const faceHeight = Math.max(landmarkDistance(observation.landmarks, 10, 152, width, height), 0.001);
  const jawWidth = landmarkDistance(observation.landmarks, 172, 397, width, height);
  const noseWidth = landmarkDistance(observation.landmarks, 98, 327, width, height);
  const mouthWidth = landmarkDistance(observation.landmarks, 61, 291, width, height);
  const lowerFace = landmarkDistance(observation.landmarks, 2, 152, width, height);
  const chinWidth = landmarkDistance(observation.landmarks, 148, 377, width, height);
  const localFrame = coordinateFrame(observation.landmarks, width, height);
  const symmetryPairs: Array<[number, number]> = [[33, 263], [133, 362], [70, 300], [105, 334], [61, 291], [172, 397], [136, 365]];
  const pairedDeviation = symmetryPairs.reduce((sum, [right, left]) => {
    const rightPoint = { x: observation.landmarks[right].x * width, y: observation.landmarks[right].y * height };
    const leftPoint = { x: observation.landmarks[left].x * width, y: observation.landmarks[left].y * height };
    return sum + Math.abs(localFrame.local(rightPoint).v - localFrame.local(leftPoint).v) / faceHeight;
  }, 0) / symmetryPairs.length * 100;

  return {
    pose,
    semanticLandmarkCount: SEMANTIC_LANDMARK_COUNT,
    measurementCount: MEASUREMENT_CATALOG.length,
    validMeasurementCount: measurements.filter((measurement) => measurement.validity === "valid").length,
    overallConfidence,
    expression: {
      label: expression.label,
      confidence: expression.confidence,
      blockedRegions: expression.blockedRegions,
    },
    regionEditability,
    measurements,
    unavailableMeasurements: [
      "true hairline / trichion",
      "ears / tragion",
      "true bony zygion or gonion",
      "calibrated millimetres",
      "maxillary or cheekbone projection hidden by soft tissue",
    ],
    metrics: {
      jawToFace: jawWidth / faceWidth,
      noseToFace: noseWidth / faceWidth,
      mouthToFace: mouthWidth / faceWidth,
      lowerThird: lowerFace / faceHeight,
      pairedDeviation,
      mouthToNose: mouthWidth / Math.max(noseWidth, 0.001),
      chinToJaw: chinWidth / Math.max(jawWidth, 0.001),
      faceAspect: faceHeight / faceWidth,
    },
  };
}

function mixUnit(value: number) {
  return clamp(value > 1 ? value / 100 : value);
}

export function createMorphPlan(analysis: FaceAnalysis, requestedMix: DirectionMix): MorphPlan {
  const mix = {
    harmony: mixUnit(requestedMix.harmony),
    symmetry: mixUnit(requestedMix.symmetry),
    dimorphism: mixUnit(requestedMix.dimorphism),
  };
  const rejectedReasons: string[] = [];
  const evidenceFamilies = buildPlannerEvidence(analysis);
  const evidenceMeasurementCount = new Set(
    evidenceFamilies.flatMap((family) => family.validMeasurementIds),
  ).size;
  const actionMap = new Map<MorphPrimitive, PlannedAction>();
  const editability = new Map(analysis.regionEditability.map((entry) => [entry.region, entry]));
  const poseAllowsPlan = analysis.pose.class !== "unsupported" && analysis.overallConfidence >= 0.5;

  const propose = (
    primitive: MorphPrimitive,
    region: FaceRegion,
    amount: number,
    direction: keyof DirectionMix,
    rationale: string,
    evidenceConfidence: number,
  ) => {
    if (Math.abs(amount) < 0.012 || mix[direction] <= 0) return;
    const regionEditability = editability.get(region);
    const score = regionEditability?.score ?? 0;
    const confidence = Math.min(analysis.overallConfidence, clamp(evidenceConfidence));
    const expressionBlocked = analysis.expression.blockedRegions.includes(region);
    if (!poseAllowsPlan || !regionEditability?.editable || confidence < 0.5 || expressionBlocked) {
      rejectedReasons.push(`${region}: ${regionEditability?.reasons[0] ?? "confidence gate did not pass"}.`);
      return;
    }
    // Confidence and editability decide whether an edit is defensible. Once it
    // clears that gate, use them once as reliability instead of repeatedly
    // shrinking an otherwise valid plan. The square-root direction response
    // keeps medium slider values useful while preserving exact zero and full.
    const directionResponse = Math.sqrt(mix[direction]);
    const reliability = 0.85 + Math.sqrt(confidence * score) * 0.15;
    const weightedAmount = amount * directionResponse * reliability;
    const existing = actionMap.get(primitive);
    if (existing) {
      // Same-sign directions reinforce with a bounded union instead of a raw
      // sum. Opposing directions cancel explicitly. This lets Harmony and
      // Angularity interact without silently doubling one primitive's budget.
      existing.amount = Math.sign(existing.amount) === Math.sign(weightedAmount)
        ? Math.sign(existing.amount) * (
            1 - (1 - Math.abs(existing.amount)) * (1 - Math.abs(weightedAmount))
          )
        : (() => {
            const dominant = Math.abs(existing.amount) >= Math.abs(weightedAmount)
              ? existing.amount
              : weightedAmount;
            const opposing = dominant === existing.amount ? weightedAmount : existing.amount;
            // Two valid style directions can disagree without making the
            // entire region disappear. Preserve a coherent dominant vector,
            // while still allowing the opposing direction to soften it.
            const residual = Math.abs(dominant) - Math.abs(opposing) * 0.35;
            const dominantFloor = Math.abs(dominant) * 0.7;
            return Math.sign(dominant) * Math.max(residual, dominantFloor);
          })();
      existing.amount = clamp(existing.amount, -0.9, 0.9);
      existing.confidence = Math.min(existing.confidence, confidence);
      existing.editability = Math.min(existing.editability, score);
      existing.maxDisplacement = Math.min(existing.maxDisplacement, regionEditability?.maxDisplacement ?? existing.maxDisplacement);
      if (!existing.directions.includes(direction)) existing.directions.push(direction);
      existing.rationale = `${existing.rationale} ${rationale}`;
    } else {
      actionMap.set(primitive, {
        primitive,
        region,
        amount: clamp(weightedAmount, -0.9, 0.9),
        confidence,
        editability: score,
        maxDisplacement: regionEditability?.maxDisplacement ?? 0.01,
        directions: [direction],
        rationale,
      });
    }
  };

  if (!poseAllowsPlan) {
    rejectedReasons.push(analysis.pose.class === "unsupported" ? "The pose falls outside the safe V2 range." : "Analysis confidence was too low for a defensible morph.");
  } else {
    for (const family of evidenceFamilies) {
      if (family.status !== "supported" || mix[family.direction] <= 0) continue;
      propose(
        family.primitive,
        family.region,
        family.signal * family.maxInfluence,
        family.direction,
        `${family.label}: ${family.rationale}`,
        family.confidence,
      );
    }

    if (analysis.pose.class !== "frontal" && mix.symmetry > 0) {
      rejectedReasons.push("Symmetry was disabled for this pose because perspective can imitate paired drift.");
    }

    for (const family of evidenceFamilies) {
      if (
        family.direction === "harmony" &&
        mix.harmony > 0 &&
        ["insufficient-evidence", "evidence-conflict"].includes(family.status)
      ) {
        rejectedReasons.push(`${family.region}: ${family.rationale}`);
      }
    }
  }

  const actions = [...actionMap.values()]
    .filter((action) => Math.abs(action.amount) >= 0.008)
    .sort((a, b) => Math.abs(b.amount) * b.confidence * b.editability - Math.abs(a.amount) * a.confidence * a.editability)
    .slice(0, 6);
  const averageConfidence = actions.length
    ? actions.reduce((sum, action) => sum + Math.sqrt(action.confidence * action.editability), 0) / actions.length
    : 0;
  const selectedCandidate: MorphPlan["selectedCandidate"] = !actions.length
    ? "identity"
    : averageConfidence >= 0.88
      ? "full"
      : averageConfidence >= 0.72
        ? "balanced"
        : "light";
  const moved = new Set(actions.map((action) => action.region));
  const preservedRegions = ["Cheeks and maxilla", "Eyes", "Skin texture", "Hair", "Background", ...analysis.regionEditability.map((entry) => entry.region)]
    .filter((region, index, all) => !moved.has(region as FaceRegion) && all.indexOf(region) === index)
    .slice(0, 8);
  if (!actions.length && !rejectedReasons.length) rejectedReasons.push("All pose-valid measurements stayed inside their uncertainty-expanded bands, so the no-op candidate won.");
  return {
    schemaVersion: PLANNER_SCHEMA_VERSION,
    actions,
    evidenceFamilies,
    evidenceMeasurementCount,
    preservedRegions,
    rejectedReasons: [...new Set(rejectedReasons)],
    candidateCount: 4,
    selectedCandidate,
    directionContributions: {
      harmony: Math.round(mix.harmony * 100),
      symmetry: Math.round(mix.symmetry * 100),
      dimorphism: Math.round(mix.dimorphism * 100),
    },
  };
}

// The evidence planner remains the personalized base. This editor-facing layer
// guarantees that a non-zero control also has a bounded visible effect, so a
// face already inside every broad evidence band does not make the UI feel inert.
export function createInteractiveMorphPlan(analysis: FaceAnalysis, requestedMix: DirectionMix): MorphPlan {
  const plan = createMorphPlan(analysis, requestedMix);
  if (analysis.pose.class === "unsupported") return plan;
  const mix = {
    harmony: mixUnit(requestedMix.harmony),
    symmetry: mixUnit(requestedMix.symmetry),
    dimorphism: mixUnit(requestedMix.dimorphism),
  };
  const editorialIntensity = clamp(
    (Math.min(mix.harmony, mix.symmetry, mix.dimorphism) - 0.65) / 0.35,
  );
  const editability = new Map(analysis.regionEditability.map((item) => [item.region, item]));
  const actions = new Map(plan.actions.map((action) => [action.primitive, { ...action }]));
  const add = (
    primitive: MorphPrimitive,
    region: FaceRegion,
    amount: number,
    direction: keyof DirectionMix,
    maxDisplacement: number,
    rationale: string,
  ) => {
    if (Math.abs(amount) < 0.01) return;
    const regionState = editability.get(region);
    if (!regionState?.editable || analysis.expression.blockedRegions.includes(region)) return;
    const existing = actions.get(primitive);
    actions.set(primitive, {
      primitive,
      region,
      amount: clamp((existing?.amount ?? 0) + amount, -0.9, 0.9),
      confidence: existing?.confidence ?? analysis.overallConfidence,
      editability: regionState.score,
      maxDisplacement: Math.max(existing?.maxDisplacement ?? 0, maxDisplacement),
      directions: [...new Set([...(existing?.directions ?? []), direction])],
      rationale: existing ? `${existing.rationale} ${rationale}` : rationale,
    });
  };

  add("jaw-width", "Jaw", -0.7 * mix.harmony, "harmony", 0.065, "Interactive facial refinement.");
  add("nose-width", "Nose", -0.62 * mix.harmony, "harmony", 0.032, "Interactive feature refinement.");
  add("mouth-width", "Lips", 0.3 * mix.harmony, "harmony", 0.03, "Interactive feature balance.");
  add("jaw-width", "Jaw", -0.3 * mix.dimorphism, "dimorphism", 0.065, "Interactive lower-face definition.");
  add("chin-length", "Chin", 0.82 * mix.dimorphism, "dimorphism", 0.05, "Interactive chin definition.");
  add("brow-height", "Brows", -0.58 * mix.dimorphism, "dimorphism", 0.024, "Interactive brow definition.");
  if (analysis.pose.class === "frontal") {
    add("paired-alignment", "Symmetry", 0.9 * mix.symmetry, "symmetry", 0.022, "Interactive paired alignment.");
  }
  // When every direction is deliberately high, blend them as one editorial
  // look rather than three unrelated corrections.
  add("jaw-width", "Jaw", -0.18 * editorialIntensity, "harmony", 0.045, "Editorial contour blend.");
  add("chin-length", "Chin", 0.18 * editorialIntensity, "dimorphism", 0.035, "Editorial lower-face blend.");
  add("nose-width", "Nose", -0.12 * editorialIntensity, "harmony", 0.022, "Editorial center-face blend.");
  add("mouth-width", "Lips", 0.08 * editorialIntensity, "harmony", 0.022, "Editorial feature balance.");
  add("brow-height", "Brows", -0.12 * editorialIntensity, "dimorphism", 0.016, "Editorial eye-frame definition.");

  plan.actions = [...actions.values()]
    .filter((action) => Math.abs(action.amount) >= 0.01)
    .sort((a, b) => Math.abs(b.amount) - Math.abs(a.amount));
  plan.selectedCandidate = plan.actions.length ? "full" : "identity";
  plan.editorialIntensity = editorialIntensity;
  return plan;
}

export function semanticOverlayIndices(landmarkCount: number) {
  return [...new Set(SEMANTIC_LANDMARKS.flatMap((definition) => {
    if (definition.meshIndex < landmarkCount) return [definition.meshIndex];
    return definition.fallbackIndices?.filter((index) => index < landmarkCount) ?? [];
  }))];
}
