# Harmonia V2

Public app: https://iuphysics.github.io/harmonia/

Harmonia is a private, browser-based portrait reshaping studio. It analyzes one
clear face, proposes an adaptive combination of Harmony, Symmetry, and
Angularity directions, then moves only pixels already present in the source
image. It does not generate or replace skin, hair, facial hair, shadows, color,
backgrounds, or missing anatomy.

Harmonia reports geometric relationships and planning confidence. It does not
calculate an attractiveness score, diagnose a person, or claim that one set of
ratios is a universal ideal.

## V2 analysis and planning

- MediaPipe's full 468-vertex face mesh remains the detection and deformation
  substrate.
- A versioned Harmonia schema maps 104 product-owned semantic proxies onto that
  dense mesh. These labels make planning explainable; they are not official
  MediaPipe anatomical landmarks and are not a medical annotation set.
- An exact 408-entry measurement catalog covers facial envelope, eyes, brows,
  nose, lips, lower face, jaw, chin, paired drift, and pose/profile context.
- The `3.0.0-beta.1` evidence planner no longer reads the legacy eight-value
  summary to choose edits. It evaluates versioned evidence families directly
  from the catalog; the clean frontal regression fixture currently consults 97
  unique pose-valid distance, ratio, angle, shape, and symmetry measurements.
- A Harmony proposal needs agreement from at least two independent normalized
  cue groups plus regional contour checks. Median reliability, conservative
  deviation magnitude, uncertainty bands, and conflict rejection keep one
  extreme measurement from authorizing or amplifying an edit.
- Frontal symmetry uses all 40 paired residuals and requires drift in at least
  three independent facial roles. Profile plans exclude bilateral symmetry and
  use only visible-silhouette profile evidence.
- Every measurement declares which poses can use it, its confidence inputs, and
  whether it is a target, guardrail, context signal, or preservation signal.
- Region editability combines pose, visibility, image quality, expression, mesh
  support, and distance from protected boundaries. Low-confidence regions lock
  or receive a smaller, explicit displacement budget. Current hard handle
  ceilings are 4.5% for jaw, 3.5% for chin and face envelope, 2.2% for nose and
  lips, 1.6% for brows, and 1.5% for symmetry alignment. A separate 4% whole-face
  displacement limit bounds the final dense result.
- Expression locks require corroborating image geometry. In particular, a high
  detector pucker coefficient does not by itself lock the chin and lips when
  mouth-to-face and mouth-to-nose widths still describe neutral closed lips.
- Harmony, Symmetry, and Angularity are blended into one joint plan instead of
  being applied as mutually exclusive filters. The default mix is `70 / 25 / 55`.
- The visible global Strength control is the only user-facing final amplitude
  control: it starts at `100`, while `0` is the unchanged source. The renderer
  maps normalized slider position `s` through `(s × m)^2.5`, where the verified
  pose calibration `m` is `0.80` and smoothly reserves up to `0.25` more through
  sliver-sensitive mild yaw. This uses the full slider without the former
  high-end safety cliff. Planner labels such as light, balanced, and full
  describe evidence; they do not silently reduce the rendered edit.
- Frontal and three-quarter controls use a pose-stable scale derived from the
  larger of projected face width and `0.65 ×` visible face height. Profiles stay
  on a more conservative projected-width path because hidden-side mesh vertices
  can nearly overlap in 2D.
- Deadbands, edit budgets, compatibility checks, and geometry validation allow
  a stronger supported edit while still returning a no-op when an edit is not
  sufficiently supported.

The 408-entry catalog contains 210 declared ratios; it is not 408 independent
beauty rules. Only calibrated, pose-valid evidence families can create edit
pressure today. The remaining measurements provide confidence, preservation,
and geometry context while additional families are validated.

The detailed contract, including primitives, schemas, pose validity, confidence,
editability, planner rules, and limitations, is in
[docs/facial-analysis-spec.md](docs/facial-analysis-spec.md).

## Capture support

Harmonia is designed for one unobstructed face in:

- a frontal view;
- a moderate three-quarter view; or
- a clean side profile with enough visible contour.

Neutral expressions and mild smiles are preferred. V2 can plan conservative
changes to the jaw, chin, nose, lips, brows, face envelope, and mild bilateral
drift. A direction or measurement is disabled when the current pose does not
show the anatomy needed to estimate it reliably; for example, bilateral
symmetry is not treated as measurable from a profile.

Choosing a new local image resets the comparison preview to **After**, so the
newly rendered result is shown immediately instead of inheriting a previous
**Before** selection.

## Pixel-only morph pipeline

1. Freeze one exact camera frame and detect its dense landmarks, optional facial
   transform, and expression signals.
2. Reject or limit frames that are blurry, poorly lit, moving, strongly posed,
   covered, or too expressive.
3. Resolve pose and map the dense mesh to 104 semantic planning proxies.
4. Evaluate the pose-valid subset of the 408-measurement catalog with confidence
   and editability scores.
5. Create and rank a bounded joint plan from the three direction weights;
   preserve regions that are already coherent or uncertain.
6. Smooth selected targets across the fixed face topology while holding an
   expanded outer boundary ring fixed at `1.18×` the face-oval radius in x and
   `1.16×` in y.
7. For a non-profile jaw, build both a continuous lower-face image-space field
   and the conservative grouped-control field. Validate them independently;
   near-front views prefer the continuous field, while other supported views
   use the safe candidate with the broader P95 landmark impact. Profiles retain
   the visible-silhouette grouped path.
8. Validate each affected region against the geometry already accepted. Reject
   non-finite displacement weights or target coordinates. If one region exceeds
   the strain budget, binary-search that region's strongest safe scale without
   shrinking unrelated regions that already passed.
9. Treat triangle foldovers and image-bound violations as hard failures at every
   strength. Non-degenerate triangles must remain within the accepted area,
   edge, stretch, and condition-number ranges. Continuous jaw candidates and
   near-frontal fields stabilize projected slivers below quality `0.02` with two
   local incident-triangle passes; they are not merged through a transitive
   whole-group average. Conservative grouped candidates use connected handling
   below `0.012`, including the profile path.
10. If a region has no safe non-zero scale, preserve its original geometry.
   Replace the accepted face region once; leave texture, lighting, color, and
   every original pixel outside the fixed warp boundary unchanged.

## Privacy

The face landmark model and MediaPipe WebAssembly runtime are bundled under
`public/`. Camera frames and local files are processed in the browser. Harmonia
does not include an image-upload API, persistence, analytics, or authentication.

## Research basis

The implementation uses the conservative, geometry-focused parts of the
published literature: MediaPipe's [468-point dense face mesh and canonical face
model](https://github.com/google-ai-edge/mediapipe/wiki/MediaPipe-Face-Mesh), the
optional [facial transformation matrix and blendshape outputs](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarkerResult),
the [Attention Mesh](https://research.google/pubs/attention-mesh-high-fidelity-face-mesh-prediction-in-real-time/)
work on locally accurate eye and lip landmarks, identity-preserving local target
changes from the [Facial Reshaping Operator](https://hubertshum.com/publications/eswa2021beautification/files/eswa2021beautification.pdf),
local shape regularization from [Laplacian surface editing](https://diglib.eg.org/items/ee43c2c6-4956-4ff0-9db1-c946306a5b99),
and foldover prevention motivated by [Locally Injective Mappings](https://igl.ethz.ch/projects/LIM/).

Normative data is treated as contextual evidence, not a universal target. The
[3D Facial Norms database](https://pubmed.ncbi.nlm.nih.gov/26492185/) provides an
example of standardized landmark and measurement collection, while a direct
[comparison of facial norm sources](https://pubmed.ncbi.nlm.nih.gov/31053285/)
found material differences between datasets. Harmonia therefore uses broad
deadbands and self-relative consistency instead of a single population mean.

No attractive-face database, attractiveness classifier, or learned personalized
planner is bundled in this release. Public research sets can inform later,
consented calibration, but their labels and populations are not universal. The
[Chicago Face Database](https://www.chicagofaces.org/) publishes standardized
face measurements and U.S. norming ratings, while
[SCUT-FBP5500](https://arxiv.org/abs/1801.06345) provides 5,500 rated frontal
portraits. Cross-cultural studies also report meaningful assessor and subject
group effects ([PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245998)),
so neither is treated as ground truth for an individual face.

## Run locally

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Validate

```bash
npm run lint
npm test
npm run build:pages
```

The suite includes a real textured-raster regression: a supported frontal plan
must create a visible lower-face pixel change while avoiding material geometric
resampling beyond the fixed face boundary.
