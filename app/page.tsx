"use client";

import { ChangeEvent, CSSProperties, DragEvent, useCallback, useEffect, useRef, useState } from "react";
import {
  analyzeFace,
  createInteractiveMorphPlan,
  DEFAULT_DIRECTION_MIX,
  detectFace,
  drawLandmarkOverlay,
  morphImage,
  type DirectionMix,
  type FaceAnalysis,
  type FaceObservation,
  type MorphPlan,
  type Point,
} from "../lib/morph";

type Status = "empty" | "camera" | "loading" | "ready" | "error";

type QualityIssue =
  | "none"
  | "framing"
  | "pixels"
  | "blur"
  | "dark"
  | "bright"
  | "yaw"
  | "roll"
  | "mouth";

type FrameAssessment = {
  ok: boolean;
  issue: QualityIssue;
  title: string;
  hint: string;
  sharpness: number;
  qualityConfidence: number;
};

type DirectionKey = "harmony" | "symmetry" | "dimorphism";

const DIRECTION_OPTIONS: Array<{
  id: DirectionKey;
  number: string;
  label: string;
  description: string;
}> = [
  {
    id: "harmony",
    number: "01",
    label: "Refine",
    description: "Slims and balances the lower face, nose, and mouth.",
  },
  {
    id: "symmetry",
    number: "02",
    label: "Balance",
    description: "Gently aligns paired features in a straight-on photo.",
  },
  {
    id: "dimorphism",
    number: "03",
    label: "Definition",
    description: "Adds a sharper jaw, chin, and brow shape.",
  },
];

const STABILITY_LANDMARKS = [10, 152, 234, 454, 33, 263, 1, 61, 291, 172, 397];

function freezeMirroredFrame(video: HTMLVideoElement) {
  const dimensions = fitDimensions(video.videoWidth, video.videoHeight, 1600);
  const canvas = document.createElement("canvas");
  canvas.width = dimensions.width;
  canvas.height = dimensions.height;
  const context = canvas.getContext("2d", { alpha: false });
  if (!context) throw new Error("Canvas is unavailable in this browser.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.translate(canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function faceCropMetrics(canvas: HTMLCanvasElement, landmarks: Point[]) {
  const xs = landmarks.map((point) => point.x * canvas.width);
  const ys = landmarks.map((point) => point.y * canvas.height);
  const minX = Math.max(0, Math.min(...xs));
  const maxX = Math.min(canvas.width, Math.max(...xs));
  const minY = Math.max(0, Math.min(...ys));
  const maxY = Math.min(canvas.height, Math.max(...ys));
  const cropWidth = Math.max(2, maxX - minX);
  const cropHeight = Math.max(2, maxY - minY);
  const sampleWidth = 224;
  const sampleHeight = Math.max(160, Math.min(300, Math.round((cropHeight / cropWidth) * sampleWidth)));
  const sample = document.createElement("canvas");
  sample.width = sampleWidth;
  sample.height = sampleHeight;
  const context = sample.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!context) throw new Error("Canvas is unavailable in this browser.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(canvas, minX, minY, cropWidth, cropHeight, 0, 0, sampleWidth, sampleHeight);
  const pixels = context.getImageData(0, 0, sampleWidth, sampleHeight).data;
  const luminance = new Float32Array(sampleWidth * sampleHeight);
  let brightness = 0;
  let darkPixels = 0;
  let brightPixels = 0;

  for (let index = 0; index < luminance.length; index += 1) {
    const offset = index * 4;
    const value = pixels[offset] * 0.299 + pixels[offset + 1] * 0.587 + pixels[offset + 2] * 0.114;
    luminance[index] = value;
    brightness += value;
    if (value <= 20) darkPixels += 1;
    if (value >= 245) brightPixels += 1;
  }

  let laplacianTotal = 0;
  let laplacianSquaredTotal = 0;
  let laplacianCount = 0;
  for (let y = 1; y < sampleHeight - 1; y += 1) {
    for (let x = 1; x < sampleWidth - 1; x += 1) {
      const index = y * sampleWidth + x;
      const laplacian =
        luminance[index - 1] +
        luminance[index + 1] +
        luminance[index - sampleWidth] +
        luminance[index + sampleWidth] -
        4 * luminance[index];
      laplacianTotal += laplacian;
      laplacianSquaredTotal += laplacian * laplacian;
      laplacianCount += 1;
    }
  }
  const laplacianMean = laplacianTotal / Math.max(1, laplacianCount);
  const sharpness = laplacianSquaredTotal / Math.max(1, laplacianCount) - laplacianMean * laplacianMean;

  return {
    sharpness,
    brightness: brightness / luminance.length,
    darkFraction: darkPixels / luminance.length,
    brightFraction: brightPixels / luminance.length,
  };
}

function assessFrame(canvas: HTMLCanvasElement, landmarks: Point[], mode: "camera" | "file"): FrameAssessment {
  const minFaceX = Math.min(...landmarks.map((point) => point.x));
  const maxFaceX = Math.max(...landmarks.map((point) => point.x));
  const minFaceY = Math.min(...landmarks.map((point) => point.y));
  const maxFaceY = Math.max(...landmarks.map((point) => point.y));
  const faceWidth = maxFaceX - minFaceX;
  const faceHeight = maxFaceY - minFaceY;
  const faceWidthPx = faceWidth * canvas.width;
  const faceHeightPx = faceHeight * canvas.height;
  const centerX = (minFaceX + maxFaceX) / 2;
  const centerY = (minFaceY + maxFaceY) / 2;
  const eyeLeft = {
    x: (landmarks[33].x + landmarks[133].x) / 2,
    y: (landmarks[33].y + landmarks[133].y) / 2,
  };
  const eyeRight = {
    x: (landmarks[263].x + landmarks[362].x) / 2,
    y: (landmarks[263].y + landmarks[362].y) / 2,
  };
  const roll = Math.abs(Math.atan2((eyeRight.y - eyeLeft.y) * canvas.height, (eyeRight.x - eyeLeft.x) * canvas.width) * (180 / Math.PI));
  const pixelDistance = (a: Point, b: Point) => Math.hypot((a.x - b.x) * canvas.width, (a.y - b.y) * canvas.height);
  const mouthWidthPx = pixelDistance(landmarks[61], landmarks[291]);
  const mouthOpen = pixelDistance(landmarks[13], landmarks[14]) / Math.max(1, mouthWidthPx);
  const mouthToFace = mouthWidthPx / Math.max(1, faceWidthPx);
  const noseToRight = pixelDistance(landmarks[1], landmarks[234]);
  const noseToLeft = pixelDistance(landmarks[1], landmarks[454]);
  const yawSignal = Math.abs(noseToRight - noseToLeft) / Math.max(1, noseToRight + noseToLeft);
  const profileLike = yawSignal > 0.38;
  const crop = faceCropMetrics(canvas, landmarks);
  const cameraMode = mode === "camera";
  const sharpnessConfidence = Math.max(0, Math.min(1, (crop.sharpness - 18) / 90));
  const exposureConfidence = Math.max(0, 1 - Math.abs(crop.brightness - 128) / 122);
  const detailConfidence = Math.max(0, Math.min(1, Math.min(faceWidthPx / 360, faceHeightPx / 460)));
  const qualityConfidence = Math.max(0, Math.min(1, 0.28 + sharpnessConfidence * 0.32 + exposureConfidence * 0.2 + detailConfidence * 0.2));
  const quality = { sharpness: crop.sharpness, qualityConfidence };

  if (cameraMode && (faceWidth < 0.23 || faceHeight < 0.34)) {
    return { ok: false, issue: "framing", title: "Move a little closer", hint: "Keep your full face inside the guide.", ...quality };
  }
  if (cameraMode && (faceWidth > 0.72 || faceHeight > 0.84)) {
    return { ok: false, issue: "framing", title: "Move a little farther back", hint: "Leave a small border around your face.", ...quality };
  }
  if (cameraMode && (centerX < 0.29 || centerX > 0.71 || centerY < 0.3 || centerY > 0.7)) {
    return { ok: false, issue: "framing", title: "Center your face", hint: "Align your eyes and chin inside the guide.", ...quality };
  }
  if (!cameraMode && (minFaceX < 0.015 || maxFaceX > 0.985 || minFaceY < 0.015 || maxFaceY > 0.985)) {
    return { ok: false, issue: "framing", title: "Face is too close to the edge", hint: "Choose a portrait with a small clear border around the face.", ...quality };
  }
  const minimumFaceWidth = cameraMode ? Math.min(240, canvas.width * 0.28) : 140;
  const minimumFaceHeight = cameraMode ? Math.min(300, canvas.height * 0.38) : 180;
  if (faceWidthPx < minimumFaceWidth || faceHeightPx < minimumFaceHeight) {
    return { ok: false, issue: "pixels", title: "Face is too small", hint: cameraMode ? "Move closer so the scan has enough detail." : "Choose a higher-resolution, closer photo.", ...quality };
  }
  if (roll > (cameraMode ? 28 : 32)) {
    return { ok: false, issue: "roll", title: "Level your head", hint: "Keep the line between your eyes roughly horizontal.", ...quality };
  }
  const mouthUnsupported = profileLike
    ? mouthOpen > (cameraMode ? 0.42 : 0.5)
    : mouthOpen > (cameraMode ? 0.2 : 0.28) || mouthToFace < (cameraMode ? 0.15 : 0.12);
  if (mouthUnsupported) {
    return { ok: false, issue: "mouth", title: "Relax your mouth", hint: "Rest your lips naturally—do not open or purse them.", ...quality };
  }
  if (crop.brightness < (cameraMode ? 48 : 36) || crop.darkFraction > (cameraMode ? 0.22 : 0.34)) {
    return { ok: false, issue: "dark", title: "Add light in front of you", hint: "Avoid a bright window behind your head.", ...quality };
  }
  if (crop.brightness > (cameraMode ? 210 : 224) || crop.brightFraction > (cameraMode ? 0.2 : 0.34)) {
    return { ok: false, issue: "bright", title: "Lighting is too bright", hint: "Step away from direct light so facial detail is visible.", ...quality };
  }
  if (crop.sharpness < (cameraMode ? 32 : 24)) {
    return { ok: false, issue: "blur", title: "Image is too blurry", hint: cameraMode ? "Wipe the lens and hold the phone still." : "Choose a sharper, in-focus portrait.", ...quality };
  }
  return { ok: true, issue: "none", title: "Pose and image quality look good", hint: "Hold still while we confirm seven clear frames.", ...quality };
}

function landmarkMotion(previous: Point[], current: Point[], width: number, height: number) {
  const previousFaceWidth = (Math.max(...previous.map((point) => point.x)) - Math.min(...previous.map((point) => point.x))) * width;
  const currentFaceWidth = (Math.max(...current.map((point) => point.x)) - Math.min(...current.map((point) => point.x))) * width;
  const faceWidth = Math.max(1, (previousFaceWidth + currentFaceWidth) / 2);
  const squaredMotion = STABILITY_LANDMARKS.reduce((total, index) => {
    const distance = Math.hypot(
      (previous[index].x - current[index].x) * width,
      (previous[index].y - current[index].y) * height,
    ) / faceWidth;
    return total + distance * distance;
  }, 0);
  return Math.sqrt(squaredMotion / STABILITY_LANDMARKS.length);
}

function fileQualityError(assessment: FrameAssessment) {
  switch (assessment.issue) {
    case "framing": return "Choose a portrait with the full face centered and a small clear border around it.";
    case "blur": return "This portrait is too blurry for a safe reshape. Choose a sharper, in-focus photo.";
    case "pixels": return "The face is too small in this portrait. Choose a higher-resolution, closer photo.";
    case "dark": return "This portrait is too dark to map safely. Choose one with soft light on the face.";
    case "bright": return "This portrait is overexposed. Choose one where facial detail is visible.";
    case "yaw": return "Choose a frontal or moderate three-quarter portrait for a reliable reshape.";
    case "roll": return "Choose a portrait with the head more level.";
    case "mouth": return "Choose a portrait with a neutral expression or mild closed-mouth smile.";
    default: return "Choose a clear, well-lit portrait with one unobstructed face.";
  }
}

function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("This image could not be opened."));
    };
    image.src = url;
  });
}

function fitDimensions(width: number, height: number, maxSide = 1800) {
  const ratio = Math.min(1, maxSide / Math.max(width, height));
  return { width: Math.round(width * ratio), height: Math.round(height * ratio) };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function firstNumber(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return undefined;
}

function firstString(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return undefined;
}

function displayLabel(value: string) {
  return value
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function poseReadout(analysis: FaceAnalysis | null) {
  const root = asRecord(analysis);
  const poseValue = root.pose;
  const pose = asRecord(poseValue);
  const rawLabel =
    (typeof poseValue === "string" ? poseValue : undefined) ??
    firstString(pose, ["label", "classification", "class", "kind", "id"]) ??
    firstString(root, ["poseClass", "poseLabel"]) ??
    "Analyzed pose";
  const rawConfidence =
    firstNumber(pose, ["confidence", "score", "poseConfidence"]) ??
    firstNumber(root, ["poseConfidence"]) ??
    0;
  const confidence = rawConfidence > 1 ? rawConfidence / 100 : rawConfidence;
  return {
    rawLabel,
    label: displayLabel(rawLabel),
    confidence: Math.max(0, Math.min(1, confidence)),
  };
}

function isUnsupportedPose(analysis: FaceAnalysis) {
  return poseReadout(analysis).rawLabel.toLowerCase().includes("unsupported");
}

function analysisCounts(analysis: FaceAnalysis | null) {
  const root = asRecord(analysis);
  const measurements = Array.isArray(root.measurements) ? root.measurements : [];
  const semanticLandmarkCount = firstNumber(root, ["semanticLandmarkCount"]) ?? 104;
  const measurementCount = firstNumber(root, ["measurementCount", "derivedMeasurementCount"]) ?? (measurements.length || 408);
  const measuredValid = measurements.filter((measurement) => asRecord(measurement).valid === true).length;
  const validMeasurementCount = firstNumber(root, ["validMeasurementCount", "validFeatureCount"]) ?? (measuredValid || measurementCount);
  return { semanticLandmarkCount, measurementCount, validMeasurementCount };
}

type EditabilityReadout = {
  region: string;
  score: number;
  reason?: string;
};

function editabilityReadouts(analysis: FaceAnalysis | null): EditabilityReadout[] {
  const root = asRecord(analysis);
  const source = root.regionEditability ?? root.editability;
  const entries: Array<[string, unknown]> = Array.isArray(source)
    ? source.map((entry, index) => {
        const item = asRecord(entry);
        return [firstString(item, ["region", "label", "id"]) ?? `region-${index}`, entry];
      })
    : Object.entries(asRecord(source));

  return entries
    .map(([key, value]) => {
      const item = asRecord(value);
      const rawScore = typeof value === "number"
        ? value
        : firstNumber(item, ["score", "editability", "confidence", "value"]) ?? 0;
      const score = rawScore > 1 ? rawScore / 100 : rawScore;
      const reasons = item.reasons;
      const reason = firstString(item, ["reason", "rationale"]) ??
        (Array.isArray(reasons) && typeof reasons[0] === "string" ? reasons[0] : undefined);
      return {
        region: displayLabel(firstString(item, ["region", "label"]) ?? key),
        score: Math.max(0, Math.min(1, score)),
        reason,
      };
    })
    .filter((item) => !item.region.toLowerCase().includes("maxilla"))
    .slice(0, 8);
}

type PlanActionReadout = {
  region: string;
  rationale: string;
  confidence?: number;
};

function planActionReadouts(plan: MorphPlan | null): PlanActionReadout[] {
  const root = asRecord(plan);
  const source = Array.isArray(root.actions)
    ? root.actions
    : Array.isArray(root.edits)
      ? root.edits
      : [];
  return source.slice(0, 4).map((value, index) => {
    if (typeof value === "string") {
      return { region: displayLabel(value), rationale: "Selected by the pose-aware planner." };
    }
    const action = asRecord(value);
    const rawConfidence = firstNumber(action, ["confidence", "score", "plannerConfidence"]);
    return {
      region: displayLabel(firstString(action, ["label", "region", "primitive", "id"]) ?? `Adjustment ${index + 1}`),
      rationale: firstString(action, ["rationale", "reason", "description"]) ?? "Pose-valid, confidence-weighted adjustment.",
      confidence: rawConfidence === undefined ? undefined : Math.max(0, Math.min(1, rawConfidence > 1 ? rawConfidence / 100 : rawConfidence)),
    };
  });
}

function plannerConfidence(plan: MorphPlan | null) {
  const root = asRecord(plan);
  const source = Array.isArray(root.actions)
    ? root.actions
    : Array.isArray(root.edits)
      ? root.edits
      : [];
  const scores = source
    .map((value) => {
      const action = asRecord(value);
      const rawConfidence = firstNumber(action, ["confidence", "score", "plannerConfidence"]);
      if (rawConfidence === undefined) return undefined;
      const rawEditability = firstNumber(action, ["editability", "regionConfidence"]);
      const confidence = Math.max(0, Math.min(1, rawConfidence > 1 ? rawConfidence / 100 : rawConfidence));
      const editability = rawEditability === undefined
        ? 1
        : Math.max(0, Math.min(1, rawEditability > 1 ? rawEditability / 100 : rawEditability));
      return Math.sqrt(confidence * editability);
    })
    .filter((value): value is number => value !== undefined);
  if (!scores.length) return null;
  return scores.reduce((total, value) => total + value, 0) / scores.length;
}

function preservedRegions(plan: MorphPlan | null) {
  const root = asRecord(plan);
  const value = root.preservedRegions;
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string").map(displayLabel).slice(0, 5);
}

function rejectedReasons(plan: MorphPlan | null) {
  const value = asRecord(plan).rejectedReasons;
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

export default function Home() {
  const originalRef = useRef<HTMLCanvasElement>(null);
  const resultRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sourceRef = useRef<HTMLCanvasElement | null>(null);
  const observationRef = useRef<FaceObservation | null>(null);
  const analysisRef = useRef<FaceAnalysis | null>(null);
  const planRef = useRef<MorphPlan | null>(null);
  const exportCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanRunRef = useRef(0);
  const [status, setStatus] = useState<Status>("empty");
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const [directionMix, setDirectionMix] = useState<DirectionMix>(DEFAULT_DIRECTION_MIX);
  const [strength, setStrength] = useState(100);
  const [showOriginal, setShowOriginal] = useState(false);
  const [showMesh, setShowMesh] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [regions, setRegions] = useState<string[]>([]);
  const [movement, setMovement] = useState(0);
  const [safetyScale, setSafetyScale] = useState(1);
  const [safetyStatus, setSafetyStatus] = useState<"passed" | "weakened" | "identity">("passed");
  const [analysis, setAnalysis] = useState<FaceAnalysis | null>(null);
  const [plan, setPlan] = useState<MorphPlan | null>(null);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanMessage, setScanMessage] = useState("Starting private scanner");
  const [scanHint, setScanHint] = useState("Keep one clear face inside the guide.");

  const stopCamera = useCallback(() => {
    scanRunRef.current += 1;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const paint = useCallback(() => {
    const source = sourceRef.current;
    const observation = observationRef.current;
    const faceAnalysis = analysisRef.current;
    const resultCanvas = resultRef.current;
    const originalCanvas = originalRef.current;
    if (!source || !observation || !faceAnalysis || !resultCanvas || !originalCanvas) return;

    const nextPlan = createInteractiveMorphPlan(faceAnalysis, directionMix);
    const transformed = morphImage(source, observation.landmarks, nextPlan, strength);
    planRef.current = nextPlan;
    exportCanvasRef.current = transformed.canvas;
    setPlan(nextPlan);
    for (const [destination, image] of [
      [resultCanvas, transformed.canvas],
      [originalCanvas, source],
    ] as const) {
      destination.width = image.width;
      destination.height = image.height;
      const context = destination.getContext("2d", { alpha: false });
      context?.drawImage(image, 0, 0);
    }
    if (showMesh) drawLandmarkOverlay(resultCanvas, observation.landmarks);
    setRegions(transformed.movedRegions);
    setMovement(transformed.maxMovementPx);
    setSafetyScale(transformed.safetyScale);
    setSafetyStatus(transformed.safetyStatus);
  }, [directionMix, strength, showMesh]);

  useEffect(() => {
    if (status !== "ready") return;
    const timer = window.setTimeout(paint, 25);
    return () => window.clearTimeout(timer);
  }, [paint, status]);

  const processFile = async (file?: File) => {
    if (!file) return;
    stopCamera();
    setShowOriginal(false);
    if (!file.type.startsWith("image/")) {
      setError("Choose a JPG, PNG, or WebP portrait.");
      setStatus("error");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("Choose an image smaller than 20 MB.");
      setStatus("error");
      return;
    }
    setStatus("loading");
    setError("");
    setFileName(file.name);
    try {
      const image = await loadImage(file);
      const dimensions = fitDimensions(image.naturalWidth, image.naturalHeight);
      const source = document.createElement("canvas");
      source.width = dimensions.width;
      source.height = dimensions.height;
      const context = source.getContext("2d", { alpha: false });
      if (!context) throw new Error("Canvas is unavailable in this browser.");
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = "high";
      context.drawImage(image, 0, 0, dimensions.width, dimensions.height);

      const observation = await detectFace(source);
      const assessment = assessFrame(source, observation.landmarks, "file");
      if (!assessment.ok) throw new Error(fileQualityError(assessment));
      const nextAnalysis = analyzeFace(observation, source.width, source.height, {
        qualityConfidence: assessment.qualityConfidence,
        temporalStability: 1,
      });
      if (isUnsupportedPose(nextAnalysis)) {
        throw new Error("This head angle is outside the safe morph range. Choose a clear front, three-quarter, or clean side profile.");
      }
      sourceRef.current = source;
      observationRef.current = observation;
      analysisRef.current = nextAnalysis;
      setAnalysis(nextAnalysis);
      setStatus("ready");
      window.setTimeout(paint, 0);
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "The portrait could not be analyzed.";
      setError(message);
      setStatus("error");
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void processFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    void processFile(event.dataTransfer.files?.[0]);
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera scanning is not supported in this browser. Choose a photo instead.");
      setStatus("error");
      return;
    }
    setError("");
    setScanProgress(8);
    setScanMessage("Starting private scanner");
    setScanHint("Keep one clear face inside the guide.");
    setStatus("camera");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30, max: 30 },
        },
      });
      streamRef.current = stream;
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const video = videoRef.current;
      if (!video) throw new Error("The camera preview could not start.");
      video.srcObject = stream;
      await video.play();
      const run = ++scanRunRef.current;
      let stableFrames = 0;
      let previousLandmarks: Point[] | null = null;
      let bestFrame: {
        canvas: HTMLCanvasElement;
        observation: FaceObservation;
        assessment: FrameAssessment;
        temporalStability: number;
      } | null = null;
      setScanProgress(20);
      setScanMessage("Finding your face");
      setScanHint("Center one front, three-quarter, or profile view.");

      const scan = async () => {
        if (run !== scanRunRef.current || !streamRef.current) return;
        try {
          if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
            setScanMessage("Warming up the camera");
            setScanHint("The first clear frame can take a moment.");
            setScanProgress(18);
            window.setTimeout(scan, 180);
            return;
          }

          // Detection runs on this frozen, mirrored canvas so the chosen pixels
          // and landmarks are guaranteed to come from the exact same frame.
          const frame = freezeMirroredFrame(video);
          const observation = await detectFace(frame);
          const landmarks = observation.landmarks;
          const assessment = assessFrame(frame, landmarks, "camera");

          if (!assessment.ok) {
            stableFrames = 0;
            bestFrame = null;
            previousLandmarks = landmarks;
            setScanMessage(assessment.title);
            setScanHint(assessment.hint);
            setScanProgress(28);
          } else {
            const motion = previousLandmarks ? landmarkMotion(previousLandmarks, landmarks, frame.width, frame.height) : Number.POSITIVE_INFINITY;
            previousLandmarks = landmarks;
            if (motion > 0.012) {
              stableFrames = 0;
              bestFrame = null;
              setScanMessage("Hold still");
              setScanHint("Keep your head and phone steady while focus locks.");
              setScanProgress(38);
            } else {
              stableFrames += 1;
              const temporalStability = Math.max(0, Math.min(1, 1 - motion * 34));
              if (!bestFrame || assessment.sharpness > bestFrame.assessment.sharpness) {
                bestFrame = { canvas: frame, observation, assessment, temporalStability };
              }
              setScanMessage(stableFrames < 7 ? `Checking clear frame ${stableFrames} of 7` : "Clear scan captured");
              setScanHint(stableFrames < 7 ? "Stay still and keep a relaxed, closed-mouth expression." : "Using the sharpest frame from this scan.");
              setScanProgress(Math.min(100, 38 + stableFrames * 9));
            }
          }

          if (stableFrames >= 7 && bestFrame) {
            const chosen = bestFrame;
            const nextAnalysis = analyzeFace(chosen.observation, chosen.canvas.width, chosen.canvas.height, {
              qualityConfidence: chosen.assessment.qualityConfidence,
              temporalStability: chosen.temporalStability,
            });
            if (isUnsupportedPose(nextAnalysis)) {
              stableFrames = 0;
              bestFrame = null;
              setScanMessage("Turn toward a supported view");
              setScanHint("Use a clear front, moderate three-quarter, or clean side profile.");
              setScanProgress(30);
              window.setTimeout(scan, 180);
              return;
            }
            sourceRef.current = chosen.canvas;
            observationRef.current = chosen.observation;
            analysisRef.current = nextAnalysis;
            setAnalysis(nextAnalysis);
            setFileName("camera-scan.jpg");
            setShowOriginal(false);
            setScanProgress(100);
            stopCamera();
            setStatus("ready");
            window.setTimeout(paint, 0);
            return;
          }
        } catch (cause) {
          const message = cause instanceof Error ? cause.message : "";
          if (!message.includes("clear face")) {
            stopCamera();
            setError("The face scanner could not start. Refresh the page or choose a photo.");
            setStatus("error");
            return;
          }
          stableFrames = 0;
          bestFrame = null;
          previousLandmarks = null;
          setScanMessage("No clear face yet");
          setScanHint("Remove obstructions and keep one face inside the guide.");
          setScanProgress(22);
        }
        window.setTimeout(scan, 180);
      };
      void scan();
    } catch (cause) {
      stopCamera();
      const denied = cause instanceof DOMException && (cause.name === "NotAllowedError" || cause.name === "SecurityError");
      setError(denied ? "Camera access was blocked. Allow camera access or choose a photo." : "The camera could not start. Choose a photo instead.");
      setStatus("error");
    }
  };

  const download = () => {
    const canvas = exportCanvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    const base = fileName.replace(/\.[^.]+$/, "") || "portrait";
    link.download = `${base}-harmonia-v2.png`;
    link.href = canvas.toDataURL("image/png", 1);
    link.click();
  };

  const reset = () => {
    setStatus("empty");
    setError("");
    setFileName("");
    setAnalysis(null);
    setPlan(null);
    setSafetyScale(1);
    setSafetyStatus("passed");
    setScanProgress(0);
    setScanMessage("Starting private scanner");
    setScanHint("Keep one clear face inside the guide.");
    stopCamera();
    sourceRef.current = null;
    observationRef.current = null;
    analysisRef.current = null;
    planRef.current = null;
    exportCanvasRef.current = null;
  };

  const pose = poseReadout(analysis);
  const counts = analysisCounts(analysis);
  const editability = editabilityReadouts(analysis);
  const planActions = planActionReadouts(plan);
  const planConfidence = plannerConfidence(plan);
  const supportedEvidenceFamilies = plan?.evidenceFamilies.filter((family) => family.status === "supported").length ?? 0;
  const evidenceMeasurementCount = plan?.evidenceMeasurementCount ?? 0;
  const preserved = preservedRegions(plan);
  const rejected = rejectedReasons(plan);
  const hasPlannedEdit = planActions.length > 0;
  const hasVisibleMorph = hasPlannedEdit && safetyStatus !== "identity" && movement >= 0.5;
  const geometryScaleLabel = !hasPlannedEdit
    ? "No edit"
    : strength === 0
      ? "Not run"
      : safetyStatus === "identity"
        ? "Blocked"
        : `${Math.round(safetyScale * 100)}%`;
  const geometryBackoffLabel = !hasPlannedEdit
    ? "Not needed"
    : strength === 0
      ? "Not run"
      : safetyStatus === "identity"
        ? "Blocked"
        : safetyStatus === "weakened"
          ? `${Math.round((1 - safetyScale) * 100)}%`
          : "None";
  const guardrailTitle = !hasPlannedEdit
    ? "No edit selected"
    : strength === 0
      ? "Strength is set to zero"
      : safetyStatus === "identity"
        ? "Mesh guardrail blocked this warp"
        : safetyStatus === "weakened"
          ? "Mesh guardrail reduced the warp"
          : "Mesh guardrail passed";
  const guardrailDetail = !hasPlannedEdit
    ? rejected[0] || "Every pose-valid measurement stayed inside its uncertainty range."
    : regions.length
      ? regions.join(" · ")
      : "The planner has an edit, but no pixels are moving at the current strength.";
  const activeDirectionLabels = DIRECTION_OPTIONS
    .filter((option) => directionMix[option.id] > 0)
    .map((option) => option.label);

  const updateDirection = (direction: DirectionKey, value: number) => {
    setShowOriginal(false);
    setDirectionMix((current: DirectionMix) => ({ ...current, [direction]: value }));
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Harmonia home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>HARMONIA</span>
        </a>
        <div className="privacy-chip"><span /> On-device processing</div>
        <nav aria-label="Application links">
          <button className="nav-button" onClick={() => document.getElementById("how")?.scrollIntoView({ behavior: "smooth" })}>How it works</button>
          <button className="nav-button" onClick={() => fileInputRef.current?.click()}>New edit</button>
        </nav>
      </header>

      {status !== "ready" && <section className="hero" id="top">
        <div className="eyebrow"><span>PIXEL-ONLY MORPHING</span><span className="eyebrow-line" /></div>
        <h1>Scan. Measure.<br /><em>Harmonize.</em></h1>
        <p className="hero-copy">Harmony, Symmetry and Angularity are blended into one personalized plan that reshapes your source pixels only.</p>
      </section>}

      <section className={`studio ${status === "ready" ? "studio-active" : ""}`} aria-label="Portrait editor">
        <div className="editor-stage">
          {status === "camera" ? (
            <div className="camera-stage">
              <video ref={videoRef} className="camera-video" muted playsInline aria-label="Live camera face scan" />
              <div className="camera-frame" aria-hidden="true"><span /><span /><span /><span /></div>
              <div className="camera-status">
                <strong>{scanMessage}</strong>
                <div className="scan-meter"><i style={{ width: `${scanProgress}%` }} /></div>
                <small>{scanHint}</small>
              </div>
              <button className="camera-cancel" onClick={() => { stopCamera(); setStatus("empty"); }}>Cancel</button>
            </div>
          ) : status === "empty" || status === "error" ? (
            <div
              className={`dropzone ${dragging ? "is-dragging" : ""}`}
              onDragEnter={() => setDragging(true)}
              onDragLeave={() => setDragging(false)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={onDrop}
            >
              <div className="face-guide" aria-hidden="true">
                <span className="guide-corner tl" /><span className="guide-corner tr" />
                <span className="guide-corner bl" /><span className="guide-corner br" />
                <div className="guide-face"><i className="guide-eye left" /><i className="guide-eye right" /><i className="guide-nose" /><i className="guide-mouth" /></div>
              </div>
              <h2>{status === "error" ? "Try again" : "Scan your face"}</h2>
              <p>{error || "The camera maps and captures your face automatically. No upload required."}</p>
              <div className="capture-actions">
                <button className="primary-button" onClick={() => void startCamera()}><span aria-hidden="true">◎</span> Start camera</button>
                <button className="secondary-button" onClick={() => fileInputRef.current?.click()}>Choose photo instead</button>
              </div>
              <small>JPG, PNG or WEBP · Up to 20 MB · Never uploaded</small>
            </div>
          ) : (
            <div className="canvas-wrap">
              <canvas ref={resultRef} className={showOriginal ? "hidden-canvas" : ""} aria-label="Morphed portrait preview" />
              <canvas ref={originalRef} className={!showOriginal ? "hidden-canvas" : ""} aria-label="Original portrait preview" />
              {status === "loading" && (
                <div className="analyzing"><span className="scanner" /><strong>Mapping facial structure</strong><small>Finding safe reshape regions…</small></div>
              )}
              <div className="compare-control" aria-label="Before and after preview">
                <button className={showOriginal ? "active" : ""} onClick={() => setShowOriginal(true)}>Before</button>
                <button className={!showOriginal ? "active" : ""} onClick={() => setShowOriginal(false)}>After</button>
              </div>
              {status === "ready" && !hasVisibleMorph && (
                <div className="no-morph-banner" role="status">No visible morph passed the current safety checks.</div>
              )}
            </div>
          )}
          {status === "loading" && (
            <div className="loading-panel"><span className="loading-orbit" /><strong>Analyzing your portrait</strong><small>The first run may take a few seconds.</small></div>
          )}
        </div>

        <aside className="control-panel">
          <div className="panel-heading">
            <div><span className="step-number">01</span><h2>Adjust your look</h2></div>
            {status === "ready" && <span className="ready-badge">Live preview</span>}
          </div>
          <p className="mode-intro">Move any slider and your After photo updates instantly. Tap Before anytime to compare.</p>
          <div className="direction-stack">
            {DIRECTION_OPTIONS.map((option) => (
              <article className={`direction-control direction-${option.id}`} key={option.id}>
                <div className="direction-heading">
                  <span className="direction-number">{option.number}</span>
                  <div><strong>{option.label}</strong><small>{option.description}</small></div>
                  <output>{directionMix[option.id]}%</output>
                </div>
                <input
                  className="direction-range"
                  style={{ "--direction-range": `${directionMix[option.id]}%` } as CSSProperties}
                  type="range"
                  min="0"
                  max="100"
                  value={directionMix[option.id]}
                  onChange={(event) => updateDirection(option.id, Number(event.target.value))}
                  aria-label={`${option.label} direction weight`}
                />
              </article>
            ))}
          </div>
          <div className="blend-note">
            <span className="blend-orbit" aria-hidden="true"><i /><i /><i /></span>
            <p><strong>One personalized plan</strong><small>Unsupported signals are downweighted for your detected angle.</small></p>
          </div>

          <div className="divider" />
          <div className="strength-row">
            <div><span className="step-number">02</span><h3>Overall change</h3></div>
            <output>{strength}%</output>
          </div>
          <input
            className="range"
            style={{ "--range": `${strength}%` } as CSSProperties}
            type="range"
            min="0"
            max="100"
            value={strength}
            onChange={(event) => { setShowOriginal(false); setStrength(Number(event.target.value)); }}
            aria-label="Adaptive edit strength"
          />
          <div className="range-labels"><span>Original</span><span>Strong</span></div>
          <p className="impact-note"><strong>High settings broaden supported changes.</strong> Foldover, feature-crossing and outer-boundary guardrails stay hard.</p>

          {status === "ready" && (
            <div className={`safety-card ${!hasPlannedEdit ? "is-no-plan" : `is-${safetyStatus}`}`}>
              <div className="safety-summary">
                <span className="safety-icon">◇</span>
                <div>
                  <strong>{guardrailTitle}</strong>
                  <small>{guardrailDetail}</small>
                </div>
              </div>
              <div className="safety-strength-flow" aria-label={`Requested strength ${strength} percent; geometry scale ${geometryScaleLabel}`}>
                <div><small>Requested strength</small><output>{strength}%</output></div>
                <span aria-hidden="true">→</span>
                <div><small>Geometry scale</small><output>{geometryScaleLabel}</output></div>
              </div>
              <div className="safety-diagnostics">
                <span><small>Planner reliability</small><b>{planConfidence === null ? "No action" : `${Math.round(planConfidence * 100)}%`}</b></span>
                <span><small>Geometry backoff</small><b>{geometryBackoffLabel}</b></span>
                <span><small>Peak movement</small><b>{hasPlannedEdit && safetyStatus !== "identity" ? `${movement.toFixed(1)} px` : "None"}</b></span>
              </div>
            </div>
          )}

          {status === "ready" && analysis && (
            <section className="intelligence-card" aria-label="Pose-aware facial analysis">
              <div className="intelligence-heading">
                <div><span className="step-number">03</span><div><strong>Face intelligence</strong><small>Structure map, never a beauty score</small></div></div>
                <span className="pose-pill"><i />{pose.label} · {Math.round(pose.confidence * 100)}%</span>
              </div>

              <div className="analysis-stats">
                <div><strong>{Math.round(counts.semanticLandmarkCount)}</strong><small>semantic anchors</small></div>
                <div><strong>{Math.round(counts.measurementCount)}</strong><small>derived measures</small></div>
                <div><strong>{Math.round(counts.validMeasurementCount)}</strong><small>valid for pose</small></div>
              </div>

              <div className="analysis-subheading"><strong>Region editability</strong><small>confidence × pose × expression</small></div>
              <div className="editability-grid">
                {editability.map((item) => (
                  <div className={`editability-chip ${item.score < 0.55 ? "is-locked" : ""}`} key={item.region} title={item.reason}>
                    <span>{item.region}</span>
                    <i><b style={{ width: `${item.score * 100}%` }} /></i>
                    <output>{item.score < 0.55 ? "LOCK" : `${Math.round(item.score * 100)}%`}</output>
                  </div>
                ))}
              </div>

              <div className="analysis-subheading plan-heading"><strong>Selected plan</strong><small>{activeDirectionLabels.join(" + ") || "No direction active"}</small></div>
              <div className="plan-evidence" aria-label={`${evidenceMeasurementCount} pose-valid evidence inputs across ${supportedEvidenceFamilies} supported families`}>
                <span><b>{evidenceMeasurementCount}</b><small>evidence inputs</small></span>
                <span><b>{supportedEvidenceFamilies}</b><small>supported families</small></span>
                <span><b>V3</b><small>robust planner</small></span>
              </div>
              <div className="plan-list">
                {planActions.length ? planActions.map((action, index) => (
                  <article key={`${action.region}-${index}`}>
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <div><strong>{action.region}</strong><small>{action.rationale}</small></div>
                    {action.confidence !== undefined && <output>{Math.round(action.confidence * 100)}%</output>}
                  </article>
                )) : (
                  <div className="identity-plan"><span>◇</span><p><strong>Preserve this geometry</strong><small>{rejected[0] || "No pose-valid deviation cleared the confidence and editability gates."}</small></p></div>
                )}
              </div>

              {!!preserved.length && <p className="preserved-note"><strong>Preserved</strong> {preserved.join(" · ")}</p>}
            </section>
          )}

          <div className="panel-actions">
            <button className="mesh-button" disabled={status !== "ready"} onClick={() => setShowMesh((value) => !value)} aria-pressed={showMesh}><span>⌘</span>{showMesh ? "Hide map" : "Show map"}</button>
            <button className="export-button" disabled={status !== "ready" || !hasVisibleMorph} onClick={download}>Export exact PNG <span>↓</span></button>
          </div>
          {status === "ready" && <button className="start-over" onClick={reset}>Start over with a different photo</button>}
        </aside>
      </section>

      {status !== "ready" && <section className="principles" id="how">
        <div className="principle-title"><span>BUILT DIFFERENT</span><h2>No generation.<br />Just careful geometry.</h2></div>
        <div className="principle-list">
          <article><span>01</span><div><h3>Your pixels stay yours</h3><p>The image is processed inside your browser. Nothing is sent to a server or retained.</p></div></article>
          <article><span>02</span><div><h3>Identity stays intact</h3><p>A bounded mesh moves existing pixels only. No new skin, hair, shadows or background.</p></div></article>
          <article><span>03</span><div><h3>Personalized, then protected</h3><p>Every adjustment responds to your own structure. Pose validity, editability and mesh safety can still weaken—or reject—a change before pixels move.</p></div></article>
        </div>
      </section>}

      <footer><span>HARMONIA / V2</span><p>A creative geometry tool. Results are subjective—not a score or measure of your worth.</p><span>PRIVATE BY DEFAULT</span></footer>
      <input ref={fileInputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
    </main>
  );
}
