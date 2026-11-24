/* MiniMax H3 Vision: the app page. It talks to the plugin's server (/mmflow/api/...),
   gets render progress over ComfyUI's websocket, and shows files through ComfyUI's /view. */

/* ------------------------------------------------------------------
   Options. The server sends the real lists; these are only fallbacks.
   ------------------------------------------------------------------ */
let OPT = {
  models: [{ name: "MiniMax H3", steps: 20, maxSteps: 40 }],
  aspects: { "16:9": [16, 9], "9:16": [9, 16], "1:1": [1, 1], "4:3": [4, 3], "3:4": [3, 4], "21:9": [21, 9] },
  megapixels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0],
  length: { min: 2, max: 20, fps: 24 },
  loras: [],
  missing: null,             // why no video model can be used, when none can
  root: "minimax_flow",
  // The Image tab's options (MOSS-Image). The server sends the real lists.
  image: {
    models: [{ id: "moss_image_sft_int8_convrot.safetensors", name: "MOSS-Image SFT (INT8)", steps: 50, minSteps: 1, maxSteps: 80, stepSize: 1, cfg: 3.0 },
      { id: "moss_image_sft_8stepdmd_int8_convrot.safetensors", name: "MOSS-Image SFT Turbo 8-step (INT8)", steps: 8, minSteps: 1, maxSteps: 80, stepSize: 1, cfg: 1.0 }],
    aspects: { "1:1": [1, 1], "4:5": [4, 5], "3:4": [3, 4], "2:3": [2, 3], "9:16": [9, 16], "16:9": [16, 9], "3:2": [3, 2], "4:3": [4, 3], "5:4": [5, 4], "21:9": [21, 9] },
    megapixels: [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    recommendedMP: [1.0, 1.5, 2.0],
    cfg: { min: 1, max: 10, step: 0.5 },
    missing: null,
  },
  // The Klein tab's options (FLUX.2 [klein] 9B). The server sends the real lists.
  klein: {
    models: [{ id: "flux-2-klein-9b.safetensors", name: "Klein 9B Distilled (BF16)", steps: 4, minSteps: 1, maxSteps: 12, stepSize: 1, cfg: 1.0, base: false }],
    aspects: { "1:1": [1, 1], "4:5": [4, 5], "3:4": [3, 4], "2:3": [2, 3], "9:16": [9, 16], "16:9": [16, 9], "3:2": [3, 2], "4:3": [4, 3], "5:4": [5, 4], "21:9": [21, 9] },
    megapixels: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    recommendedMP: [1.0, 1.5, 2.0],
    cfg: { min: 1, max: 10, step: 0.5 },
    slots: 4,
    missing: null,
  },
  // The Pruna tab's options (P-H3-Image). The server sends the real lists.
  pruna: {
    models: [{ id: "lora", name: "MiniMax H3 + P-H3-Image LoRA", steps: 8, minSteps: 8, maxSteps: 8, stepSize: 1, cfg: 1.0 }],
    aspects: { "1:1": [1, 1], "4:3": [4, 3], "3:2": [3, 2], "16:9": [16, 9], "21:9": [21, 9], "3:4": [3, 4], "2:3": [2, 3], "9:16": [9, 16] },
    megapixels: [1.0, 2.0, 4.0, 8.0],
    slots: 9, steps: 8, enhance: false,
    missing: null,
  },
};
const TONES = ["#2a3a37", "#33302a", "#2b2d3a", "#3b3328", "#26332f", "#30262a", "#222a33", "#2f2a36"];

/* ------------------------------------------------------------------
   State. Everything on screen is drawn from this.
   ------------------------------------------------------------------ */
const state = {
  ready: false,
  view: "gallery",           // "gallery" | "clip"
  openId: null,
  filter: "all",
  perRow: 3,
  prompt: "",
  caret: 0,
  mentionIndex: 0,
  mentionClosed: false,
  refs: [],                  // every uploaded image: {id, name, file, added, img, tone}
  attached: [],              // ids of the images in the prompt bar
  motion: null,              // Motion from: the id of the video that gives a render its motion (Motion Transfer)
  motionMethod: "tara",      // "tara" (Motion Transfer TARA v1.0, several people) or "guide" (Motion Transfer v1.0)
  motionWhy: { tara: null, swap: null },  // why a method can't run here (from GET swap)
  motionPeople: [],          // Motion's picked people: [{x, y, picture}], painted in PEOPLE_COLORS' order
  motionExtra: "",           // instructions added to the end of the prompt Vision writes for them
  motionMarks: false,        // whether SAM 3.1 is here to mark them
  motionTurbo: {},           // per model id that Motion can use: whether it runs its own way (a turbo build)
  pickingMotion: false,      // the next video clicked in the gallery becomes Motion from
  library: { open: false, source: "uploads", selected: null, selectedGen: null, query: "", sort: "newest" },
  clips: [],
  tab: "video",              // "video" | "image" | "klein" | "edit": "edit" is the editor (edit.js), the others the gallery and prompt
  negative: "",              // the Image tab's negative prompt (empty: the model's recommended one)
  negativeOpen: false,
  drafts: { video: "", image: "", klein: "", kleinSwap: "", pruna: "", prunaSheet: "", edit: "" },   // each tab keeps its own prompt
  videoSettings: { model: "HyperFlow H3 Flash", steps: 6, res: 0.6, aspect: "16:9", len: 5, seedMode: "random", seed: 48213, loras: [] },
  imageSettings: { model: "MOSS-Image SFT (INT8)", steps: 50, cfg: 3.0, res: 1.0, aspect: "1:1", seedMode: "random", seed: 48213, loras: [] },
  // size: "match" (image 1's shape) or an aspect ratio, which is also kept in aspect.
  kleinSettings: { model: "Klein 9B Distilled (BF16)", steps: 4, cfg: 1.0, res: 1.0, size: "match", aspect: "1:1", seedMode: "random", seed: 48213, loras: [],
    // Face swap: its size, the BFS LoRA, Only the head (a painted mask) with its sampler, Refine face, the mask's edge.
    swapRes: 2.0, swapLora: "", headOnly: false, swapSampler: "standard", lanSteps: 20, refine: false, refineStrength: 0.5, maskGrow: 8, maskFeather: 12 },
  // The Klein tab's four slots, image 1-4: ids of images from the Add image library, or null. The slots keep
  // their names: whatever is in slot 3 is "image 3" in the prompt.
  kleinSlots: [null, null, null, null],
  // Klein's Face swap: picture 1 the body and scene, picture 2 the face (the head to put on it).
  kleinMode: "edit",
  kleinSwapSlots: [null, null, null, null, null],
  // The head mask painted on the body picture: { image: its id, mask: the file on the server, url }.
  kleinSwapMask: null,
  // The Pruna tab (P-H3-Image): Image (slots picture 1-9) or Character sheet (the character, then up to three
  // pictures that help with its face, outfit or details, and the views to draw). res is Image's size, sheetRes the sheet's.
  prunaSettings: { model: "MiniMax H3 + P-H3-Image LoRA", steps: 8, res: 4.0, sheetRes: 8.0, size: "match", aspect: "1:1", seedMode: "random", seed: 48213, loras: [], enhance: false },
  prunaMode: "image",
  prunaSlots: Array(9).fill(null),
  prunaSheetSlots: [null, null, null, null],
  prunaSheet: { preset: "standard", views: ["head_front", "head_side", "body_front", "body_side", "body_back"], roles: ["character", "face", "outfit", "detail"], bg: "white" },
  // Grid selection, for Delete: ids of the tiles that are ticked, and the last one clicked (for Shift ranges).
  picked: new Set(),
  lastPicked: null,
  // The Image tab's Edit switch: one picture to edit (an Add image library id) with MOSS-Image's edit LoRA.
  imageEdit: false,
  imageEditSlot: [null],
  settingsOpen: false,
  loraPicker: false,
};
// state.settings is the settings of the tab you're on.
Object.defineProperty(state, "settings", {
  get: () => (state.tab === "image" ? state.imageSettings : state.tab === "klein" ? state.kleinSettings : state.tab === "pruna" ? state.prunaSettings : state.videoSettings),
});
const deleted = new Set();   // clips deleted here, so a late progress message can't bring one back

/* ------------------------------------------------------------------
   Helpers
   ------------------------------------------------------------------ */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const newSeed = () => Math.floor(Math.random() * 2 ** 31);
const clipById = (id) => state.clips.find((c) => c.id === id);
const mpText = (mp) => (Math.round(mp * 10) === mp * 10 ? Number(mp).toFixed(1) : String(mp)) + " MP";
const isImage = () => state.tab === "image";
const isKlein = () => state.tab === "klein";
const isPruna = () => state.tab === "pruna";
const sheetOn = () => isPruna() && state.prunaMode === "sheet";
const swapOn = () => isKlein() && state.kleinMode === "swap";
// The tabs whose results are pictures, and the tabs with named slots (Klein's image 1-4, Pruna's picture 1-9).
const imageTab = () => isImage() || isKlein() || isPruna();
const isSlotTab = () => isKlein() || isPruna();
// Where the prompt box's text is kept: each tab, and Pruna's sheet notes apart from its image prompt.
const draftKey = () => (sheetOn() ? "prunaSheet" : swapOn() ? "kleinSwap" : state.tab);
// The Image tab with Edit on: one slot, the picture to edit.
const editOn = () => isImage() && state.imageEdit;
// The tabs with picture slots in the prompt bar: Klein (image 1-4) and the Image tab's Edit (one picture).
const hasSlots = () => isSlotTab() || editOn();
const slots = () => (isKlein() ? (swapOn() ? state.kleinSwapSlots : state.kleinSlots) : isPruna() ? (sheetOn() ? state.prunaSheetSlots : state.prunaSlots) : state.imageEditSlot);
const ROLE_NAMES = { character: "character", face: "face", outfit: "outfit", detail: "detail" };
const slotWord = (i) => (swapOn() ? (i === 0 ? "body" : i === 1 ? "face" : "face " + i) : isKlein() ? "image " + (i + 1) : sheetOn() ? (i ? ROLE_NAMES[state.prunaSheet.roles[i]] || "detail" : "character")
  : isPruna() ? "picture " + (i + 1) : "the picture to edit");
const SlotWord = (i) => slotWord(i).replace(/^./, (ch) => ch.toUpperCase());
// "image 2" (Klein) or "picture 2" (Pruna) in a prompt.
const slotRe = (tab = state.tab) => (tab === "pruna" ? /\bpicture\s?([1-9])\b/gi : /\bimage\s?([1-4])\b/gi);
const kindOf = (c) => c.kind || "video";
// Which tab a clip belongs to: Klein's images are images too (page, download, Use in a video), in their own tab.
const tabOf = (c) => (c.family === "klein" ? "klein" : c.family === "ph3" ? "pruna" : kindOf(c));
// The options of the tab you're on: models, aspect ratios and sizes.
const tabOpt = () => (isImage() ? OPT.image : isKlein() ? OPT.klein : isPruna() ? OPT.pruna : { models: OPT.models, aspects: OPT.aspects, megapixels: OPT.megapixels });
// With nothing installed, the settings keep the saved model so the pop-up still works.
const modelOf = (name) => {
  const m = tabOpt().models.find((x) => x.name === name) || tabOpt().models[0]
    || { name: name || "Not installed", steps: state.settings.steps || 20, minSteps: 1, maxSteps: 80, stepSize: 1, cfg: state.settings.cfg || 3 };
  // Editing uses the edit LoRA's settings, whatever the checkpoint.
  const e = OPT.image.edit;
  return editOn() && e ? { ...m, steps: e.steps, cfg: e.cfg, minSteps: e.minSteps, maxSteps: e.maxSteps, stepSize: 1, sampler: e.sampler, scheduler: e.scheduler, edit: true } : m;
};
const aspectOf = (a) => OPT.aspects[a] || OPT.image.aspects[a] || OPT.pruna.aspects[a] || (/^\d+:\d+$/.test(a || "") ? a.split(":").map(Number) : [16, 9]);
// A clip's shape: its aspect ratio, or (Klein's Match image 1) its exact width and height.
const shapeOf = (c) => ((c.family === "klein" || c.family === "ph3") && c.width && c.height ? c.width + ":" + c.height : c.aspect);
const clock = (s) => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
const toneOf = (id) => TONES[[...id].reduce((a, ch) => a + ch.charCodeAt(0), 0) % TONES.length];
const viewUrl = (folder, file) => `/view?type=output&subfolder=${encodeURIComponent(OPT.root + "/" + folder)}&filename=${encodeURIComponent(file)}`;
const fileUrl = (subfolder, file) => `/view?type=output&subfolder=${encodeURIComponent(subfolder)}&filename=${encodeURIComponent(file)}`;
const clipUrl = (c) => c.url || (c.file ? fileUrl(c.subfolder || OPT.root + (kindOf(c) === "image" ? "/stills" : "/clips"), c.file) : "");
// H3 takes lengths in steps of 17 frames, so a clip gets the nearest such length.
const snapFrames = (seconds) => 5 + 17 * Math.max(1, Math.round((seconds * OPT.length.fps - 5) / 17));
const isBusy = (c) => c.status === "queued" || c.status === "rendering";
const loraText = (l) => l.name.replace(/\\/g, "/").split("/").pop().replace(/\.[^.]+$/, "") + " · " + Number(l.strength).toFixed(2);
const attach = (id) => { if (!state.attached.includes(id)) state.attached.push(id); };

// Width and height for an aspect ratio at a size in megapixels. H3 wants sides in steps of 32, MOSS-Image in steps of 16.
function dims(aspect, mp, kind = state.tab) {
  if (kind === "video") return videoDims(aspect, mp);
  const [a, b] = aspectOf(aspect);
  const grid = kind === "image" || kind === "klein" ? 16 : 32;
  const pixels = mp * 1024 * 1024;
  const w = Math.max(grid, Math.round(Math.sqrt((pixels * a) / b) / grid) * grid);
  const h = Math.max(grid, Math.round(Math.sqrt((pixels * b) / a) / grid) * grid);
  return [w, h];
}
const sizeText = (aspect, mp) => dims(aspect, mp).join("×");
// MiniMax H3's sizes, as flow_server.video_dims works them out: 1.0 MP is H3's own size (1344x768 at 16:9, as
// large as its pixel limit allows in each shape); up to it, sizes stay within that limit; and a size that would
// come out more than 3% the wrong shape is picked for its shape.
const H3_LIMIT = 1344 * 768;
const H3_NATIVE = { "16:9": [1344, 768], "9:16": [768, 1344], "1:1": [992, 992], "4:3": [1152, 864], "3:4": [864, 1152], "21:9": [1536, 672] };
function videoDims(aspect, mp) {
  if (Math.abs(mp - 1) < 1e-6 && H3_NATIVE[aspect]) return H3_NATIVE[aspect];
  const [a, b] = aspectOf(aspect), ratio = a / b, pixels = mp * 1024 * 1024, g = 32, cap = mp <= 1 ? H3_LIMIT : 0;
  // Python's round() rounds halves to even; the same here, so both always agree.
  const rnd = (x) => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : f % 2 ? f + 1 : f; };
  let w = Math.max(g, rnd(Math.sqrt(pixels * ratio) / g) * g), h = Math.max(g, rnd(Math.sqrt(pixels / ratio) / g) * g);
  if (Math.abs(w / h / ratio - 1) <= 0.03 && (!cap || w * h <= cap)) return [w, h];
  let best = null;
  for (let hh = g; hh < 4096; hh += g) {
    for (const ww of new Set([Math.max(g, Math.floor((hh * ratio) / g) * g), Math.ceil((hh * ratio) / g) * g])) {
      if (cap && ww * hh > cap) continue;
      const area = Math.abs((ww * hh) / pixels - 1), shape = Math.abs(ww / hh / ratio - 1);
      if (area > 0.12) continue;
      const score = shape + 0.35 * area;
      if (!best || score < best[0] - 1e-12) best = [score, ww, hh];
    }
  }
  return [best[1], best[2]];
}

async function api(method, path, body) {
  const form = body instanceof FormData;
  const res = await fetch("/mmflow/api/" + path, {
    method,
    headers: body && !form ? { "Content-Type": "application/json" } : undefined,
    body: form ? body : body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error((await res.text()) || res.statusText);
  return res.json();
}

const ICON = {
  x: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  close: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  plus: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 5v14M5 12h14"/></svg>',
  back: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M19 12H5M11 6l-6 6 6 6"/></svg>',
  star: (on) => `<svg width="16" height="16" viewBox="0 0 24 24" fill="${on ? "currentColor" : "none"}" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 3l2.8 5.7 6.2.9-4.5 4.4 1.1 6.2L12 17.3 6.4 20.2l1.1-6.2L3 9.6l6.2-.9z"/></svg>`,
  dice: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="4"/><circle cx="8.5" cy="8.5" r="1.2" fill="currentColor"/><circle cx="15.5" cy="15.5" r="1.2" fill="currentColor"/><circle cx="12" cy="12" r="1.2" fill="currentColor"/></svg>',
  play: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4l13 8-13 8z"/></svg>',
  pause: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14"/><rect x="14" y="5" width="4" height="14"/></svg>',
  expand: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
  shrink: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>',
  search: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></svg>',
  upload: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg>',
  download: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 4v12M7 11l5 5 5-5M5 20h14"/></svg>',
  trash: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>',
  swap: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7 4 3 8l4 4"/><path d="M3 8h13a5 5 0 0 1 0 10h-2"/><path d="m17 20 4-4-4-4"/></svg>',
  face: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M9 10h.01M15 10h.01M8.5 15a5 5 0 0 0 7 0"/></svg>',
  reuse: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M4 12a8 8 0 0 1 14-5.3L20 9M20 4v5h-5M20 12a8 8 0 0 1-14 5.3L4 15M4 20v-5h5"/></svg>',
  video: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3"/></svg>',
  image: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/></svg>',
  check: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M5 12l5 5 9-10"/></svg>',
  pruna: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M12 3l1.8 4.7L18.5 9.5l-4.7 1.8L12 16l-1.8-4.7L5.5 9.5l4.7-1.8z"/><path d="M18 15l.9 2.1L21 18l-2.1.9L18 21l-.9-2.1L15 18l2.1-.9z"/></svg>',
  klein: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="8" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/><path d="M17 13v8M13 17h8"/></svg>',
  edit: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 15h18M8 15v4M14 15v4"/></svg>',
  copy: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/></svg>',
};

function mentionsIn(text) {
  return [...text.matchAll(/@([A-Za-z0-9_]+)/g)].map((m) => m[1]);
}

// The prompt as HTML, with known @names highlighted and unknown ones underlined.
function promptHTML(text, kind = state.tab, slots = kind === "pruna" ? state.prunaSlots : state.kleinSlots, pictureCount = null) {
  if (kind === "image" || kind === "prunaSheet") return esc(text);
  // Klein: "image 1" to "image 4" name the slots, Pruna "picture 1" to "picture 9"; a slot with nothing in it is underlined.
  if (kind === "klein" || kind === "pruna") {
    const split = kind === "pruna" ? /\b(picture\s?[1-9])\b/i : /\b(image\s?[1-4])\b/i;
    return text.split(split).map((part, i) => i % 2
      ? `<span class="${slots[Number(part.slice(-1)) - 1] ? "m" : "bad"}">${esc(part)}</span>` : esc(part)).join("");
  }
  const names = new Set(state.refs.map((r) => r.name));
  // Video: <Picture N> is marked when there is a picture N (count: a clip's own, else the prompt bar's).
  const count = pictureCount ?? state.attached.length;
  return text.split(/(@[A-Za-z0-9_]+|<Picture \d+>)/).map((part) => {
    if (/^@[A-Za-z0-9_]+$/.test(part)) return `<span class="${names.has(part.slice(1)) ? "m" : "bad"}">${esc(part)}</span>`;
    const n = /^<Picture (\d+)>$/.exec(part);
    if (n) return `<span class="${Number(n[1]) >= 1 && Number(n[1]) <= count ? "m" : "bad"}">${esc(part)}</span>`;
    return esc(part);
  }).join("");
}

// A box with the clip's aspect ratio, fitted inside a 16:9 frame.
function picStyle(aspect, tone) {
  const [a, b] = aspectOf(aspect);
  const fit = a / b >= 16 / 9 ? "width:100%" : "height:100%";
  return `${fit};aspect-ratio:${a} / ${b};background:${tone}`;
}

const activeLoras = () => state.settings.loras.filter((l) => l.on);
const stepsText = (n, rec) => n + (n === rec ? " · recommended" : "");

function summary() {
  const s = state.settings;
  const n = activeLoras().length;
  if (isKlein()) {
    const m = modelOf(s.model), [w, h] = kleinSize();
    return [s.model, s.steps + " steps", ...(m.base ? ["Guidance " + s.cfg] : []), s.size === "match" ? (swapOn() ? "Match the body" : "Match image 1") : s.aspect,
      w + "×" + h, s.seedMode === "fixed" ? "Seed " + s.seed : "Random seed", n + (n === 1 ? " LoRA" : " LoRAs")].join(" · ");
  }
  if (isPruna()) {
    const [w, h] = prunaSize();
    return [s.model === OPT.pruna.models[0]?.name && OPT.pruna.models.length < 2 ? "P-H3-Image" : s.model,
      sheetOn() ? "Sheet " + sheetLayout().ratioText : s.size === "match" ? (state.prunaSlots[0] ? "Match picture 1" : "1:1") : s.aspect,
      w + "×" + h, s.seedMode === "fixed" ? "Seed " + s.seed : "Random seed", ...(n && modelOf(s.model).loras !== false ? [n + (n === 1 ? " LoRA" : " LoRAs")] : [])].join(" · ");
  }
  if (editOn()) {
    const size = editSize();
    return ["Edit", s.model, s.steps + " steps", "Guidance " + s.cfg, size ? size.join("×") : mpText(s.res),
      s.seedMode === "fixed" ? "Seed " + s.seed : "Random seed", n + (n === 1 ? " LoRA" : " LoRAs")].join(" · ");
  }
  if (!isImage() && state.motion && motionSource(state.motion))
    return ["Motion from a video (" + ({ tara: "TARA", guide: "v1.0", swap: "Character Swap" })[motionMethodNow()] + ")", modelOf(s.model)?.scheduler === "hyperflow" ? s.model + " (can't do Motion: pick another model)" : s.model, mpText(s.res), "the video's shape and length", s.seedMode === "fixed" ? "Seed " + s.seed : "Random seed",
      n + (n === 1 ? " LoRA" : " LoRAs")].join(" · ");
  return [s.model, s.steps + " steps", ...(isImage() ? ["Guidance " + s.cfg] : []), mpText(s.res), s.aspect, ...(isImage() ? [] : [s.len + " s"]),
    s.seedMode === "fixed" ? "Seed " + s.seed : "Random seed", n + (n === 1 ? " LoRA" : " LoRAs")].join(" · ");
}

function progressText(c) {
  if (c.status === "queued") return "Waiting…";
  if (c.status === "failed") return "Render failed";
  if (!c.step) return c.enhance ? "Enhancing the prompt, loading the model…" : "Loading the model…";
  if (c.step >= c.steps) return kindOf(c) === "image" ? "Finishing the image…" : "Finishing the video…";
  return "Step " + c.step + " of " + c.steps;
}
const barWidth = (c) => ((c.step || 0) / c.steps) * 100 + "%";
const takeProgress = (c) => (c.status === "queued" ? " · waiting" : c.status === "rendering" ? " · step " + (c.step || 0) + "/" + c.steps : c.status === "failed" ? " · failed" : "");

let toastTimer;
function say(text) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 3200);
}

// Settings, the prompt draft and the prompt bar are saved on the server, so they survive a reload.
let saveTimer;
function saveUi() {
  if (!state.ready) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    state.drafts[draftKey()] = state.prompt;
    api("PUT", "ui", { settings: state.videoSettings, imageSettings: state.imageSettings, attached: state.attached, motion: state.motion, motionMethod: state.motionMethod, motionPeople: state.motionPeople, motionExtra: state.motionExtra, perRow: state.perRow,
      prompt: state.drafts.video, imagePrompt: state.drafts.image, imageNegative: state.negative, negativeOpen: state.negativeOpen,
      kleinSettings: state.kleinSettings, kleinPrompt: state.drafts.klein, kleinSlots: state.kleinSlots,
      kleinMode: state.kleinMode, kleinSwapSlots: state.kleinSwapSlots, kleinSwapPrompt: state.drafts.kleinSwap, kleinSwapMask: state.kleinSwapMask,
      prunaSettings: state.prunaSettings, prunaPrompt: state.drafts.pruna, prunaSheetPrompt: state.drafts.prunaSheet, prunaMode: state.prunaMode,
      prunaSlots: state.prunaSlots, prunaSheetSlots: state.prunaSheetSlots, prunaSheet: state.prunaSheet,
      imageEdit: state.imageEdit, imageEditSource: state.imageEditSlot[0],
      tab: state.tab }).catch(() => {});
  }, 400);
}

/* ------------------------------------------------------------------
   Drawing
   ------------------------------------------------------------------ */
function render() {
  const editing = state.view === "gallery" && state.tab === "edit";
  const inGallery = state.view === "gallery" && !editing;
  if (state.view === "gallery") { $("pv")?.pause(); shownClip = null; }
  $("gallery").hidden = !inGallery;
  $("dock").hidden = !inGallery;
  if (editing) {
    $("clipView").hidden = true;
    $("scrim").hidden = true;
    renderHeader();
    EDIT.show();
    saveUi();
    return;
  }
  EDIT.hide();
  $("clipView").hidden = inGallery;
  // An open pop-up darkens everything else: the page, the top bar and the prompt bar, all but its own button.
  const popup = inGallery && (state.settingsOpen || state.library.open);
  $("scrim").hidden = !popup;
  $("composerDim").hidden = !popup;
  document.body.classList.toggle("popup-open", popup);
  document.body.classList.toggle("library-open", popup && state.library.open);
  renderHeader();
  if (inGallery) {
    renderFilters();
    renderGrid();
    renderRefs();
    renderPrompt();
    renderSettings();
    renderLibrary();
  } else {
    renderClip();
  }
  saveUi();
}

function renderHeader() {
  $("tabs").hidden = state.view !== "gallery";
  $("tabs").innerHTML = [["video", "Video", ICON.video], ["image", "Image", ICON.image], ["klein", "Klein 9b", ICON.klein], ["pruna", "Pruna", ICON.pruna], ["edit", "Edit", ICON.edit]].map(([id, label, icon]) => {
    const busy = state.clips.filter((c) => tabOf(c) === id && isBusy(c)).length;
    return `<button class="tab ${state.tab === id ? "on" : ""}" data-act="tab" data-id="${id}" role="tab" aria-selected="${state.tab === id}">${icon}${label}${busy ? `<span class="tab-busy" title="${busy} rendering">${busy}</span>` : ""}</button>`;
  }).join("");
  $("headLeft").innerHTML = (state.view === "clip"
    ? `<button class="back" data-act="back">${ICON.back}Back to gallery</button>`
    : `<span class="brand">MiniMax H3 Vision</span>`)
    + (state.ready && missingWhy() && state.view === "gallery" && state.tab !== "edit" ? `<button class="missing" data-act="whyMissing" title="${esc(missingWhy())}">Not installed</button>` : "");
  const busy = state.clips.filter(isBusy).length;
  $("queueDot").classList.toggle("on", busy > 0);
  $("queueText").textContent = busy ? busy + " rendering" : "Queue empty";
}

// Why this tab can't render, when its models aren't installed.
const missingWhy = () => (isImage() ? OPT.image.missing : isKlein() ? OPT.klein.missing : isPruna() ? OPT.pruna.missing : OPT.missing) || "";

const tabClips = () => state.clips.filter((c) => tabOf(c) === state.tab);
// Starred is hidden for now (the stars are kept with each clip): the grid shows everything.
const filteredClips = () => tabClips();

function renderFilters() {
  const n = pickedHere().length, what = imageTab() ? (n === 1 ? "image" : "images") : n === 1 ? "video" : "videos";
  $("filters").innerHTML = `<span class="chip on count">All · ${tabClips().length}</span>`
    + `<label class="chip import" title="Bring in a video or image made earlier. Its prompt and settings are read from the file. A .zip from Export .zip brings its reference images too. You can also drop files on the page.">${ICON.upload}Import
        <input type="file" accept="video/*,image/png,image/jpeg,image/webp,.zip,application/zip" multiple data-change="importFiles"></label>`
    + `<button class="chip delete-chip" data-act="deletePicked" ${n ? "" : "disabled"} title="${n ? `Delete the ${n} ticked ${what}` : "Tick videos or images to delete them: the box on each, or ⌘-click (Ctrl-click) and Shift-click"}">${ICON.trash}Delete${n ? " " + n : ""}</button>`
    + (n ? `<button class="link clear-pick" data-act="clearPicked">Clear selection</button>` : "");
}

function pickClip(id, e) {
  const list = filteredClips().map((c) => c.id);
  if (e && e.shiftKey && state.lastPicked && list.includes(state.lastPicked)) {
    const [a, b] = [list.indexOf(state.lastPicked), list.indexOf(id)].sort((x, y) => x - y);
    list.slice(a, b + 1).forEach((x) => state.picked.add(x));
  } else if (state.picked.has(id)) state.picked.delete(id);
  else state.picked.add(id);
  state.lastPicked = id;
  renderFilters(); renderGrid();
}

function showAppMenu(open) {
  $("appMenu").hidden = !open;
  $("closeBtn").setAttribute("aria-expanded", String(open));
  if (open) $("appMenu").querySelector(".app-menu-item").focus();
}

// Yes/no in the app's own dialog. danger: a red Yes (for Delete).
let confirmAnswer = null;
function askConfirm(title, text, yes, danger = false) {
  $("confirmTitle").textContent = title;
  $("confirmText").textContent = text;
  $("confirmYes").textContent = yes;
  $("confirmYes").classList.toggle("danger", danger);
  $("confirmYes").classList.toggle("primary", !danger);
  $("confirm").hidden = false;
  (danger ? $("confirmNo") : $("confirmYes")).focus();
  return new Promise((resolve) => { confirmAnswer = resolve; });
}
function answerConfirm(yes) {
  $("confirm").hidden = true;
  const done = confirmAnswer;
  confirmAnswer = null;
  if (done) done(yes);
}

// The ticked tiles that are in this tab and still exist.
const pickedHere = () => tabClips().filter((c) => state.picked.has(c.id)).map((c) => c.id);

function renderGrid() {
  const grid = $("grid");
  grid.style.setProperty("--per-row", state.perRow);
  grid.classList.toggle("picking", pickedHere().length > 0);
  $("perRow").value = state.perRow;
  $("perRowVal").textContent = state.perRow;
  $("perRowLabel").textContent = imageTab() ? "Images per row" : "Videos per row";
  const clips = filteredClips();
  if (!clips.length) {
    tileKeys = new WeakMap();
    grid.innerHTML = `<div class="empty">${state.ready ? (state.filter === "starred" ? (isImage() ? "No starred images yet." : "No starred clips yet.") : imageTab() ? (sheetOn() ? "No sheets yet. Add a character picture below, pick the views and press the arrow button." : "No images yet. Describe one below and press the arrow button.") : "Nothing here yet. Write a prompt below and press the arrow button.")
      : state.loadError ? `The MiniMax H3 Vision server answered with an error, so nothing could be loaded. Trying again…<br><span class="fail-why">${esc(state.loadError.slice(0, 300))}</span>` : "Loading…"}</div>`;
    return;
  }
  patchGrid(grid, clips.map((c) => [c.id, kindOf(c) === "image" ? imageTile(c) : videoTile(c)]));
  // A kept tile's progress (Waiting… → Loading the model… → Step n of m) is brought up to date here.
  clips.forEach((c) => {
    if (!isBusy(c)) return;
    const tile = grid.querySelector(`[data-clip="${c.id}"]`);
    const text = tile?.querySelector("[data-progtext]"), fill = tile?.querySelector("[data-prog]");
    if (text && text.textContent !== progressText(c)) text.textContent = progressText(c);
    if (fill) fill.style.width = barWidth(c);
  });
  if (hover.id) requestAnimationFrame(hoverFrame);
}

// Tiles are only rebuilt when what they show changed. Rebuilding every tile reloaded every video, so
// opening or closing a pop-up made the whole grid blink. Progress text and bar width are left out of
// the comparison: renderGrid sets them on the kept tiles.
let tileKeys = new WeakMap();
const tileKey = (html) => html.replace(/(data-progtext="[^"]*">)[^<]*/, "$1").replace(/(data-prog="[^"]*" style=")[^"]*/, "$1");
function patchGrid(grid, tiles) {
  const old = new Map([...grid.children].filter((el) => el.dataset.clip).map((el) => [el.dataset.clip, el]));
  const nodes = tiles.map(([id, html]) => {
    const el = old.get(id), key = tileKey(html);
    if (el && tileKeys.get(el) === key) return el;
    const t = document.createElement("template");
    t.innerHTML = html.trim();
    const fresh = t.content.firstElementChild;
    tileKeys.set(fresh, key);
    return fresh;
  });
  // Leave the grid alone when nothing changed; otherwise put the tiles in order, moving only what moved.
  const now = [...grid.children];
  if (now.length === nodes.length && nodes.every((n, i) => now[i] === n)) return;
  [...grid.children].forEach((el) => { if (!nodes.includes(el)) el.remove(); });
  nodes.forEach((n, i) => { if (grid.children[i] !== n) grid.insertBefore(n, grid.children[i] || null); });
}

// A video in the grid.
function videoTile(c) {
  const busy = isBusy(c), failed = c.status === "failed";
  return `<div class="tile ${state.picked.has(c.id) ? "picked" : ""}" data-clip="${c.id}">
      <div class="frame">
        <span class="pic" style="${picStyle(c.aspect, toneOf(c.id))}">${busy && c.preview ? liveVideo(c) : c.file ? `<video src="${clipUrl(c)}" muted loop playsinline preload="metadata"></video>` : ""}</span>
        <button class="tile-hit" data-act="open" data-id="${c.id}" aria-label="Open clip: ${esc(c.prompt)}"></button>
        ${busy ? `<span class="busy ${c.preview ? "has-live" : ""}"><span data-progtext="${c.id}">${progressText(c)}</span><span class="track"><span class="fill" data-prog="${c.id}" style="width:${barWidth(c)}"></span></span>
          <button class="cancel" data-act="cancel" data-id="${c.id}">Cancel</button></span>` : ""}
        ${failed ? `<span class="busy failed"><span>Render failed</span><span class="fail-why">${esc(c.error || "")}</span>
          <button class="cancel" data-act="deleteClip" data-id="${c.id}">Remove</button></span>` : ""}
        <span class="len" data-len="${c.id}">${c.len} s</span>${c.faces ? `
        <span class="badge" title="Faces refined from another clip">Faces refined</span>` : ""}
        ${c.file ? `<span class="hover-line" aria-hidden="true"><span data-hoverbar="${c.id}"></span></span>` : ""}
        <button class="pick ${state.picked.has(c.id) ? "on" : ""}" data-act="pick" data-id="${c.id}" role="checkbox" aria-checked="${state.picked.has(c.id)}" aria-label="Select for Delete" title="Select (⌘-click or Shift-click the tiles for more)">${ICON.check}</button>
      </div>
    </div>`;
}

// An image in the grid: the picture, or its live preview while it renders.
function imageTile(c) {
  const busy = isBusy(c), failed = c.status === "failed", done = c.status === "done";
  const picture = busy && c.preview ? `<img data-live="${c.id}" src="${c.preview}" alt="">` : done ? `<img src="${clipUrl(c)}" alt="" loading="lazy" draggable="false">` : "";
  // A finished image can be dragged: onto a Klein slot, or into the Video tab's prompt as an @image.
  return `<div class="tile ${state.picked.has(c.id) ? "picked" : ""}" data-clip="${c.id}" ${done ? `draggable="true" data-drag-gen="${c.id}"` : ""}>
      <div class="frame">
        <span class="pic" style="${picStyle(shapeOf(c), toneOf(c.id))}">${picture}</span>
        <button class="tile-hit" data-act="open" data-id="${c.id}" aria-label="Open image: ${esc(c.prompt)}"></button>
        ${busy ? `<span class="busy ${c.preview ? "has-live" : ""}"><span data-progtext="${c.id}">${progressText(c)}</span><span class="track"><span class="fill" data-prog="${c.id}" style="width:${barWidth(c)}"></span></span>
          <button class="cancel" data-act="cancel" data-id="${c.id}">Cancel</button></span>` : ""}
        ${failed ? `<span class="busy failed"><span>Render failed</span><span class="fail-why">${esc(c.error || "")}</span>
          <button class="cancel" data-act="deleteClip" data-id="${c.id}">Remove</button></span>` : ""}
        <span class="len">${c.mode === "sheet" ? "Sheet" : (c.family === "klein" || c.family === "ph3") && c.width ? c.width + "×" + c.height : c.aspect}</span>
        ${done && isSlotTab() ? `<button class="edit-this" data-act="editThis" data-id="${c.id}" title="Put this image in ${isPruna() ? "picture 1" : "image 1"} and keep editing it">Edit this</button>` : ""}
        <button class="pick ${state.picked.has(c.id) ? "on" : ""}" data-act="pick" data-id="${c.id}" role="checkbox" aria-checked="${state.picked.has(c.id)}" aria-label="Select for Delete" title="Select (⌘-click or Shift-click the tiles for more)">${ICON.check}</button>
      </div>
    </div>`;
}

// The Image tab's negative prompt: a second, smaller box under the prompt, opened from the bar.
function renderNegative() {
  const on = isImage();
  $("negToggle").hidden = !on;
  const e = OPT.image.edit;
  $("editToggle").hidden = !on;
  $("editToggle").classList.toggle("on", editOn());
  $("editToggle").setAttribute("aria-pressed", editOn());
  $("editToggle").disabled = !(e && e.available) && !editOn();
  $("editToggle").title = e && e.available
    ? (editOn() ? "Editing a picture with the MOSS-Image edit LoRA. Click to make new images again." : "Edit one picture with the MOSS-Image edit LoRA: describe only the change.")
    : `Edit needs the MOSS-Image edit LoRA (${(e && e.lora) || "moss_image_edit_lora_rank_64_bf16.safetensors"}) in models/loras.`;
  $("editToggle").innerHTML = (editOn() ? "− " : "+ ") + "Edit" + (editOn() ? " · on" : "");
  $("negBox").hidden = !(on && state.negativeOpen);
  if (!on) return;
  $("negToggle").innerHTML = (state.negativeOpen ? "− " : "+ ") + "Negative prompt" + (state.negative.trim() ? "" : " · none");
  $("negToggle").setAttribute("aria-expanded", state.negativeOpen);
  const box = $("negative");
  if (box.value !== state.negative) box.value = state.negative;
  box.placeholder = "No negative prompt";
}

// The Pruna tab's parts of the prompt bar: the Image / Character sheet switch, the sheet's views, the Enhance
// prompt switch and "What the model gets".
function renderPrunaBar() {
  const on = isPruna();
  $("prunaMode").hidden = !on && !isKlein();
  $("sheetViews").hidden = !sheetOn();
  $("enhanceToggle").hidden = !on;
  $("prunaSent").hidden = !on && !swapOn();
  if (isKlein()) {
    // Klein: Edit (image 1-4) or Face swap (body and face, with the BFS LoRA).
    $("prunaMode").innerHTML = `<div class="seg">${[["edit", "Edit"], ["swap", "Face swap"]].map(([id, label]) =>
      `<button class="${state.kleinMode === id ? "on" : ""}" data-act="kleinMode" data-id="${id}" aria-pressed="${state.kleinMode === id}">${label}</button>`).join("")}</div>
      <span class="hint">${swapOn() ? (OPT.klein.swap ? "Puts the head from the face picture onto the body picture, with the BFS head-swap LoRA."
        : "Face swap needs the BFS head-swap LoRA (bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors) in models/loras.")
        : "Edit up to four pictures, or make one from the prompt."}</span>`;
    if (swapOn()) $("prunaSentText").textContent = swapText();
    // Face swap's switches: Only the head (needs the head painted) and Refine face (needs Only the head).
    $("enhanceToggle").hidden = !swapOn();
    $("refineToggle").hidden = !swapOn();
    if (swapOn()) {
      const k = state.kleinSettings, body = state.kleinSwapSlots[0], painted = !!maskFor(body), ho = k.headOnly && painted;
      $("enhanceToggle").innerHTML = (ho ? "− " : "+ ") + "Only the head" + (ho ? " · on" : "");
      $("enhanceToggle").classList.toggle("on", ho); $("enhanceToggle").setAttribute("aria-pressed", ho); $("enhanceToggle").disabled = !body;
      $("enhanceToggle").title = "Paint the head on the body picture: only that part changes, everything else stays exactly as it was.";
      const rf = k.refine && ho;
      $("refineToggle").innerHTML = (rf ? "− " : "+ ") + "Refine face" + (rf ? " · on" : "");
      $("refineToggle").classList.toggle("on", rf); $("refineToggle").setAttribute("aria-pressed", rf); $("refineToggle").disabled = !ho;
      $("refineToggle").title = ho ? "A second pass on the head at 1024 px for sharper detail, pasted back in." : "Refine face works with Only the head: paint the head first.";
    }
    return;
  }
  if (!on) return;
  $("prunaMode").innerHTML = `<div class="seg">${[["image", "Image"], ["sheet", "Character sheet"]].map(([id, label]) =>
    `<button class="${state.prunaMode === id ? "on" : ""}" data-act="prunaMode" data-id="${id}" aria-pressed="${state.prunaMode === id}">${label}</button>`).join("")}</div>
    <span class="hint">${sheetOn() ? "A sheet of one character from several sides, drawn from its picture." : "Make a picture, or edit picture 1 with up to eight more as references."}</span>`;
  const pm = modelOf(state.prunaSettings.model);
  $("enhanceToggle").hidden = h3img(pm);   // H3-Image reads the prompt as written
  const en = state.prunaSettings.enhance && !h3img(pm);
  $("enhanceToggle").innerHTML = (en ? "− " : "+ ") + "Enhance prompt" + (en ? " · on" : "");
  $("enhanceToggle").classList.toggle("on", en);
  $("enhanceToggle").setAttribute("aria-pressed", en);
  $("enhanceToggle").disabled = !OPT.pruna.enhance && !en;
  $("enhanceToggle").title = OPT.pruna.enhance
    ? "Runs Pruna's prompt upsampler first: it rewrites your text into a long, detailed prompt for the model (a few seconds more). Off, your text goes to the model as it is."
    : "Enhance prompt needs Pruna's prompt upsampler (p-h3-image_prompt_upsampler_bf16.safetensors) in models/text_encoders.";
  const sent = prunaSent();
  $("prunaSentText").textContent = sent ? sent + (en ? "\n\n→ Pruna's upsampler rewrites this first. What it wrote is on the picture's page afterwards." : "")
    : sheetOn() ? "Pick at least one view." : "Write a prompt.";
  if (!sheetOn()) return;
  const sh = state.prunaSheet, L = sheetLayout();
  const row = (label, keys) => `<div class="view-row"><span class="view-label">${label}</span>${keys.map((k) =>
    `<button class="chip ${sh.views.includes(k) ? "on" : ""}" data-act="sheetView" data-id="${k}" aria-pressed="${sh.views.includes(k)}">${SHEET_VIEWS[k][1]}</button>`).join("")}</div>`;
  // A small drawing of the sheet: heads (two to a column when stacked), tall full-body panels, a close-up column.
  const u = 34, cell = (w) => `<i style="width:${w}px"></i>`;
  let lay = "";
  if (L.stacked) for (let c = 0; c < Math.ceil(L.heads.length / 2); c++) lay += `<span class="col" style="width:${0.5 * u}px"><i></i>${L.heads[c * 2 + 1] ? "<i></i>" : '<i class="none"></i>'}</span>`;
  else lay += L.heads.map(() => cell(0.62 * u)).join("");
  lay += L.bodies.map(() => cell(0.46 * u)).join("");
  if (L.close.length) lay += `<span class="col" style="width:${0.5 * u}px">${L.close.map(() => "<i></i>").join("")}</span>`;
  $("sheetViews").innerHTML = `
    <div class="view-row"><span class="view-label">Preset</span>${Object.entries(SHEET_PRESETS).map(([k, p]) =>
      `<button class="chip preset ${sh.preset === k ? "on" : ""}" data-act="sheetPreset" data-id="${k}">${p.name}</button>`).join("")}
      <span class="sheet-layout" title="The sheet's layout: ${L.ratioText}">${lay}</span></div>
    ${row("Head", ["head_front", "head_3q", "head_side", "head_back"])}
    ${row("Full body", ["body_front", "body_3q", "body_side", "body_back"])}
    ${row("Close-ups", ["hands", "feet"])}`;
}

function renderRefs() {
  renderNegative();
  renderPrunaBar();
  // Reset (top right of the prompt) shows when there is something to clear.
  $("resetPrompt").hidden = !(state.prompt.trim() || (hasSlots() ? slots().some(Boolean) : !isImage() && state.attached.length));
  if (hasSlots()) { renderSlots(); return; }
  document.querySelector(".composer").classList.toggle("no-refs", isImage());
  // MOSS-Image takes no reference pictures, so the Image tab has no image strip.
  $("refs").hidden = isImage();
  if (isImage()) { $("hint").textContent = "MOSS-Image is good at words in pictures: put the text you want in quotes"; return; }
  // Video: the pictures in order, <Picture 1> first. The prompt is sent as written: it names them and defines
  // its own subjects. Drag a picture onto another to change the order, or onto the text to write its tag.
  const strip = state.attached.map((id) => state.refs.find((r) => r.id === id)).filter(Boolean);
  const named = [...new Set([...state.prompt.matchAll(/<Picture (\d+)>/g)].map((m) => Number(m[1])))];
  const over = named.filter((n) => n < 1 || n > strip.length);
  $("hint").innerHTML = over.length
    ? `<span class="bad-hint">${over.map((n) => "&lt;Picture " + n + "&gt;").join(", ")}: there ${strip.length === 1 ? "is 1 picture" : "are " + strip.length + " pictures"}</span>`
    : "Type @ to name a picture";
  $("refs").innerHTML = strip.map((r, i) => `
    <div class="ref ${named.includes(i + 1) ? "used" : ""}">
      <span class="ref-pic" style="background:${r.tone}" draggable="true" data-drag-ref="${r.id}" data-replace-ref="${r.id}" data-act="showRef" data-id="${r.id}" title="${esc(r.name)} · click to see it in Add image. Drag it onto the text to write &lt;Picture ${i + 1}&gt;, or onto another picture to change the order."><img src="${r.img}" alt="" draggable="false">
        <button class="ref-x" data-act="removeRef" data-id="${r.id}" aria-label="Take Picture ${i + 1} out of the prompt bar">${ICON.x}</button>
      </span>
      <button class="ref-name pic-tag" data-act="insertPicture" data-id="${i}" title="Put &lt;Picture ${i + 1}&gt; in the prompt at the cursor">Picture ${i + 1}</button>
    </div>`).join("") + `
    <button class="add-ref" id="addRef" data-act="openLibrary" aria-expanded="${state.library.open}" ${strip.length >= 9 ? "disabled" : ""}>${ICON.plus}Add image</button>` + motionChip();
}

// Motion from: a video whose motion, timing, camera and sound a render takes (Alissonerdx's Motion Transfer v1.0).
// The pictures and the prompt say who and what performs it.
const motionSource = (id) => { const c = clipById(id); return c && kindOf(c) === "video" && c.status === "done" && c.file ? c : null; };
function motionChip() {
  const c = state.motion && motionSource(state.motion);
  if (!c) return `<button class="add-ref motion-add ${state.pickingMotion ? "on" : ""}" data-act="pickMotion" title="Take the motion of a video in the gallery: your pictures and prompt say who performs it">${ICON.swap}${state.pickingMotion ? "Click a video…" : "Motion from…"}</button>`;
  return `<div class="motion-chip" title="This render takes the motion, timing, camera and sound of this video (Motion Transfer)">
      <video src="${clipUrl(c)}" muted loop autoplay playsinline></video>
      <span class="motion-text"><b>Motion from this video</b>
        <select class="motion-method" data-change="motionMethod" aria-label="How the motion is transferred" title="${esc(motionHint())}">
          <option value="tara" ${motionMethodNow() === "tara" ? "selected" : ""} ${state.motionWhy.tara ? "disabled" : ""}>TARA · several people</option>
          <option value="guide" ${motionMethodNow() === "guide" ? "selected" : ""}>Motion Transfer v1.0</option>
          <option value="swap" ${motionMethodNow() === "swap" ? "selected" : ""} ${state.motionWhy.swap ? "disabled" : ""}>Character Swap · marked people</option>
        </select>
        <button class="link" data-act="openPeople">${state.motionPeople.length ? state.motionPeople.length + (state.motionPeople.length === 1 ? " person" : " people") + " picked" + (state.motionExtra.trim() ? " + extra" : "") + " · change" : "Pick the people"}</button>
        ${state.motionPeople.length ? "" : `<button class="link" data-act="motionTemplate">Use the author's prompt</button>`}</span>
      <button class="ref-x motion-x" data-act="clearMotion" aria-label="Stop taking the motion of this video">${ICON.x}</button>
    </div>`;
}
// Pick the people: click each person on the video's first frame and say which picture they become. Each is painted
// in a colour through the clip (SAM 3), and with an empty prompt Vision writes the author's prompt for them.
const PEOPLE_COLORS = [["red", "#DC463C"], ["blue", "#2D7DE1"], ["green", "#32B45A"], ["yellow", "#EBCD23"]];
const PEOPLE = { check: null };
// Which clip and clicks a check was made for: reopening keeps it while they're the same.
const peopleKey = () => state.motion + ":" + state.motionPeople.map((p) => p.x + "," + p.y).join(";");
function openPeople() {
  const c = motionSource(state.motion);
  if (!c) return;
  if (!state.motionMarks) { say("Picking the people needs SAM 3.1: gear menu › Models › Swap and Motion."); return; }
  $("ppFrame").src = clipUrl(c);
  $("ppExtra").value = state.motionExtra;
  if (PEOPLE.check && PEOPLE.check.key !== peopleKey()) PEOPLE.check = null;
  drawPeople();
  $("peopleModal").hidden = false;
}
const stripCount = () => state.attached.filter((id) => state.refs.some((r) => r.id === id)).length;
function ppFrameRect() {
  const v = $("ppFrame"), w = v.clientWidth, h = v.clientHeight;
  const scale = Math.min(w / (v.videoWidth || w), h / (v.videoHeight || h));
  const pw = (v.videoWidth || w) * scale, ph = (v.videoHeight || h) * scale;
  return { left: (w - pw) / 2, top: (h - ph) / 2, width: pw, height: ph };
}
function drawPeople() {
  const n = stripCount(), r = ppFrameRect();
  $("ppDots").innerHTML = state.motionPeople.map((p, i) => `<span class="pp-dot" style="left:${r.left + p.x * r.width}px;top:${r.top + p.y * r.height}px;background:${PEOPLE_COLORS[i][1]}">${i + 1}</span>`).join("");
  $("ppList").innerHTML = state.motionPeople.length ? state.motionPeople.map((p, i) => `
    <div class="pp-row"><span class="pp-swatch" style="background:${PEOPLE_COLORS[i][1]}"></span><b>${PEOPLE_COLORS[i][0][0].toUpperCase() + PEOPLE_COLORS[i][0].slice(1)} person</b> becomes
      <select data-change="peoplePicture" data-index="${i}" aria-label="Picture for the ${PEOPLE_COLORS[i][0]} person">${Array.from({ length: Math.max(n, 1) }, (_, k) =>
        `<option value="${k + 1}" ${p.picture === k + 1 ? "selected" : ""}>Picture ${k + 1}</option>`).join("")}</select>
      <button class="ref-x pp-x" data-act="removePerson" data-id="${i}" aria-label="Remove the ${PEOPLE_COLORS[i][0]} person">${ICON.x}</button></div>`).join("")
    : `<p class="hint">Click a person on the frame above.</p>`;
  $("ppHint").textContent = !n ? "Add the pictures to the prompt bar first: Picture 1, Picture 2…"
    : state.motionPeople.length >= PEOPLE_COLORS.length ? "That's the most: 4 people." : "Click each person to replace, up to 4. Everyone else in the video stays as they are.";
  const ck = PEOPLE.check;
  $("ppCheck").disabled = !state.motionPeople.length || ck?.state === "working";
  $("ppCheckOut").innerHTML = !ck ? "" : ck.state === "working" ? `<span class="hint">Marking the people through the clip…</span>`
    : ck.state === "failed" ? `<span class="hint warn">${esc(ck.error || "It didn't work.")}</span>`
    : `<video src="${ck.url}" autoplay muted loop playsinline></video><span class="hint">Each colour is one person. Wrong one, or lost on the way? Remove them and click again.</span>`;
}
async function checkPeople() {
  const key = peopleKey();
  PEOPLE.check = { state: "working", key };
  drawPeople();
  try {
    const { prompt_id } = await api("POST", "clips/" + state.motion + "/swap-mark", { people: state.motionPeople, res: state.settings.res });
    PEOPLE.check.pid = prompt_id;
    while (PEOPLE.check?.pid === prompt_id) {
      await new Promise((r) => setTimeout(r, 2000));
      const r = await api("GET", "swap-mark/" + prompt_id);
      if (PEOPLE.check?.pid !== prompt_id || r.state === "working") continue;
      PEOPLE.check = r.state === "done" ? { state: "done", key, url: fileUrl(r.subfolder, r.file) } : { state: "failed", key, error: r.error };
      drawPeople();
    }
  } catch (error) { PEOPLE.check = { state: "failed", key, error: error.message }; drawPeople(); }
}
$("ppExtra").addEventListener("input", (e) => { state.motionExtra = e.target.value; saveUi(); });
$("ppFrameBox").addEventListener("click", (e) => {
  if (state.motionPeople.length >= PEOPLE_COLORS.length) return;
  const box = $("ppFrame").getBoundingClientRect(), r = ppFrameRect();
  const x = (e.clientX - box.left - r.left) / r.width, y = (e.clientY - box.top - r.top) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  const used = new Set(state.motionPeople.map((p) => p.picture));
  const free = Array.from({ length: stripCount() }, (_, k) => k + 1).find((k) => !used.has(k)) || 1;
  state.motionPeople = [...state.motionPeople, { x: Math.round(x * 10000) / 10000, y: Math.round(y * 10000) / 10000, picture: free }];
  PEOPLE.check = null;
  drawPeople(); saveUi();
});

// With Motion from a video, some Video settings are the recipe's or the video's: the settings say which.
function motionNote(model) {
  if (state.tab !== "video" || !state.motion || !motionSource(state.motion)) return "";
  const turbo = state.motionTurbo[model.id];
  const how = turbo === undefined ? "HyperFlow can't do Motion: pick another model."
    : turbo ? `${esc(model.name)} runs its own way: its own steps and sampler, without the DMD LoRA (untested with these LoRAs).`
    : `${esc(model.name)} gets the DMD 8-step turbo LoRA: 8 steps, Euler, whatever Steps says.`;
  return `<div class="motion-note"><b>Motion from a video is on.</b> ${how} Aspect ratio and Length come from the video;
    Realism and RefMods aren't used. Model, Resolution, Seed and LoRAs apply.</div>`;
}

// TARA when it can run here, else Motion Transfer v1.0.
const motionMethodNow = () => ["tara", "swap"].includes(state.motionMethod) && !state.motionWhy[state.motionMethod] ? state.motionMethod : "guide";
function motionHint() {
  return motionMethodNow() === "swap"
    ? "Character Swap v1.1 on the people you picked: each is replaced by their picture (cut out onto grey), the scene and everyone else stay. Their marks are solid colours. Needs Pick the people."
    : motionMethodNow() === "tara"
    ? "TARA: the author's LoRA for several people, objects or vehicles. The video is a native reference, kept in step frame by frame (BFS temporal alignment). Needs 2 to 15 s of video."
    : "Motion Transfer v1.0: the video traced as a guide, frame by frame. The author's default for one performer." + (state.motionWhy.tara ? " " + state.motionWhy.tara : "");
}
async function loadMotionStatus() {
  try {
    const st = await api("GET", "swap");
    state.motionWhy = { tara: st.methods?.tara || null, swap: st.methods?.swap || null };
    state.motionMarks = !!st.mark;
    state.motionTurbo = Object.fromEntries(st.models.map((m) => [m.id, m.turbo]));
  } catch {}
}
function setMotion(id) {
  if (id !== state.motion) { state.motionPeople = []; state.motionExtra = ""; }
  state.motion = id;
  state.pickingMotion = false;
  if (state.tab !== "video") ACT.tab("video");
  render(); saveUi();
  if (id) say("The next video takes this video's motion. Pick the pictures and say in the prompt who performs it.");
}

/* Klein's four slots, image 1-4. They keep their names: swapping two pictures (drag one onto another) or
   putting a new one in doesn't change the prompt. Each shows the picture's own size under it. */
const slotRef = (i, list = slots()) => state.refs.find((r) => r.id === list[i]) || null;
function refSize(r) {
  if (r.w || r.sizing) return;
  r.sizing = true;
  const im = new Image();
  im.onload = () => {
    r.w = im.naturalWidth; r.h = im.naturalHeight;
    if (hasSlots() && state.view === "gallery") { renderRefs(); $("pill").textContent = summary(); }
    if (state.view === "clip") renderClip();
  };
  im.src = r.img;
}
// The size a Klein render comes out at: image 1's shape (Match image 1), or an aspect ratio, at the resolution.
// An edit is made at the picture's own shape, scaled to the resolution in steps of 16 px. null until its size is known.
// The Video tab's size: a slider over the usual sizes, and the value can be clicked to type any size in between
// (0.98, say). 1.0 MP is H3's own size, marked on the slider.
function videoResHTML(s, T) {
  const stops = T.megapixels, at = resStop(s.res, stops), native = stops.indexOf(1.0);
  const pct = (100 * native) / (stops.length - 1);
  return `<div class="field">
      <div class="field-head"><label class="label" for="resSlide">Resolution</label>
        <button class="value value-edit ${s.res === 1 ? "rec" : ""}" id="resVal" data-act="editValue" data-id="res" title="Click to type a size, like 0.98">${resText(s)}</button></div>
      <div class="steps-wrap">
        <input id="resSlide" type="range" min="0" max="${stops.length - 1}" step="1" value="${at}" data-input="resSlide" aria-valuetext="${esc(resText(s))}">
        <span class="rec-mark" style="left:calc(${pct}% + ${(0.5 - pct / 100) * 16}px)" aria-hidden="true"></span>
      </div>
      <div class="field-head steps-foot"><span class="hint">1.0 MP is H3's own size. Click the size to type another, like 0.98.</span>
        <button class="link" data-act="nativeRes" style="visibility:${s.res === 1 ? "hidden" : "visible"}">Use 1.0</button></div>
    </div>`;
}
const resText = (s) => `${mpText(s.res)} · ${sizeText(s.aspect, s.res)}`;
// The slider's place for a size: the stop it is, or the nearest one for a typed size.
function resStop(res, stops) {
  let best = 0;
  stops.forEach((v, i) => { if (Math.abs(v - res) < Math.abs(stops[best] - res)) best = i; });
  return best;
}
// A slider's value can be clicked and typed in: Resolution (Video), Length, Steps and Guidance. What's typed is
// kept within the slider's range, and Steps snaps to the numbers the model takes.
const VALUE_FIELDS = {
  res: { id: "resVal", name: "Resolution", unit: "MP", get: (s) => s.res, set: (v) => setVideoRes(v), range: () => OPT.videoMP || { min: 0.1, max: 2 } },
  len: { id: "lenVal", name: "Length", unit: "s", get: (s) => s.len, range: () => OPT.length, whole: true,
    set: (v) => { state.settings.len = v; render(); } },
  steps: { id: "stepsVal", name: "Steps", unit: "steps", get: (s) => s.steps, whole: true,
    range: () => { const m = modelOf(state.settings.model); return { min: m.minSteps || 1, max: m.maxSteps, every: m.stepSize || 1 }; },
    set: (v, r) => { state.settings.steps = r.min + Math.round((v - r.min) / r.every) * r.every; render(); } },
  cfg: { id: "cfgVal", name: "Guidance", unit: "", get: (s) => s.cfg, range: () => (isKlein() ? OPT.klein.cfg : OPT.image.cfg) || OPT.image.cfg,
    set: (v) => { state.settings.cfg = Math.round(v * 10) / 10; render(); } },
};
function editValue(field) {
  const f = VALUE_FIELDS[field], btn = $(f.id), r = f.range();
  if (!btn) return;
  const box = document.createElement("span");
  box.className = "res-edit";
  box.innerHTML = `<input id="valueInput" type="text" inputmode="${f.whole ? "numeric" : "decimal"}" autocomplete="off" value="${f.get(state.settings)}" aria-label="${f.name} (${r.min} to ${r.max})">${f.unit ? " " + f.unit : ""}`;
  btn.replaceWith(box);
  const inp = $("valueInput");
  inp.focus(); inp.select();
  let done = false;
  const finish = (keep) => {
    if (done) return;
    done = true;
    let v = parseFloat(String(inp.value).replace(",", "."));
    inp.blur();
    if (!keep || !Number.isFinite(v)) return render();
    if (v < r.min || v > r.max) say(`${f.name}: ${r.min} to ${r.max}${f.unit && f.unit !== "steps" ? " " + f.unit : ""}.`);
    v = Math.min(r.max, Math.max(r.min, f.whole ? Math.round(v) : v));
    f.set(v, r);
  };
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    e.stopPropagation();
  });
  inp.addEventListener("blur", () => finish(true));
}
function setVideoRes(mp) {
  const lim = OPT.videoMP || { min: 0.1, max: 2 };
  mp = Math.round(Math.min(lim.max, Math.max(lim.min, mp)) * 100) / 100;
  state.settings.res = mp;
  render();
}

function editSize(res = state.imageSettings.res) {
  const r = slotRef(0, state.imageEditSlot);
  if (!r || !r.w) return null;
  const k = Math.sqrt((res * 1024 * 1024) / (r.w * r.h));
  return [Math.round((r.w * k) / 16) * 16, Math.round((r.h * k) / 16) * 16];
}

function kleinSize(s = state.kleinSettings) {
  const r = slotRef(0, swapOn() ? state.kleinSwapSlots : state.kleinSlots);
  if (swapOn()) s = { ...s, res: s.swapRes, size: "match" };
  if (s.size === "match" && r && r.w) {
    const k = Math.sqrt((s.res * 1024 * 1024) / (r.w * r.h));
    return [Math.max(16, Math.round((r.w * k) / 16) * 16), Math.max(16, Math.round((r.h * k) / 16) * 16)];
  }
  return dims(s.size === "match" ? "1:1" : s.aspect, s.res, "klein");
}
// The size a Pruna picture comes out at: picture 1's shape (Match picture 1) or an aspect ratio, in steps of 32;
// a character sheet's shape comes from its views. 1 MP is 1024×1024, as in ComfyUI's Resolution Selector.
function prunaSize(s = state.prunaSettings) {
  const fit = (w0, h0, mp) => { const k = Math.sqrt((mp * 1024 * 1024) / (w0 * h0)); return [Math.max(32, Math.round((w0 * k) / 32) * 32), Math.max(32, Math.round((h0 * k) / 32) * 32)]; };
  if (sheetOn()) return fit(sheetLayout().ratio * 1000, 1000, s.sheetRes);
  const r = slotRef(0, state.prunaSlots);
  if (s.size === "match" && r && r.w) return fit(r.w, r.h, s.res);
  return dims(s.size === "match" ? "1:1" : s.aspect, s.res, "pruna");
}

/* The Pruna tab's character sheet: the views to draw, as one row on a plain background. */
const SHEET_VIEWS = {
  head_front: ["Head", "Front", "a front-facing head-and-shoulders portrait, both eyes visible"],
  head_3q: ["Head", "¾", "a three-quarter head-and-shoulders portrait, turned 45 degrees"],
  head_side: ["Head", "Profile", "a strict side-profile head-and-shoulders portrait, one eye visible"],
  head_back: ["Head", "Back", "the back of the head and shoulders, no face visible"],
  body_front: ["Full body", "Front", "a full-body front view, neutral standing pose"],
  body_3q: ["Full body", "¾", "a full-body three-quarter view"],
  body_side: ["Full body", "Side", "a full-body side view, facing right"],
  body_back: ["Full body", "Back", "a full-body back view, no face visible"],
  hands: ["Close-up", "Hands", "a close-up of both hands"],
  feet: ["Close-up", "Feet / shoes", "a close-up of both feet or shoes"],
};
const SHEET_PRESETS = {
  turnaround: { name: "Turnaround · 4", views: ["body_front", "body_3q", "body_side", "body_back"] },
  standard: { name: "Standard · 5", views: ["head_front", "head_side", "body_front", "body_side", "body_back"] },
  full: { name: "Full · 8", views: ["head_front", "head_3q", "head_side", "head_back", "body_front", "body_3q", "body_side", "body_back"] },
  faces: { name: "Faces · 4", views: ["head_front", "head_3q", "head_side", "head_back"] },
};
const SHEET_ROLES = { face: "Face", outfit: "Outfit", detail: "Detail" };
const SHEET_BGS = ["white", "light grey", "neutral studio"];
// Heads stack two to a column when there are more than two, full-body views are tall panels and the close-ups
// share one column. ratio is the sheet's width over its height.
function sheetLayout(views = state.prunaSheet.views) {
  const heads = views.filter((v) => v.startsWith("head")), bodies = views.filter((v) => v.startsWith("body")), close = views.filter((v) => v === "hands" || v === "feet");
  const stacked = heads.length > 2;
  const ratio = Math.max(0.6, (stacked ? Math.ceil(heads.length / 2) * 0.5 : heads.length * 0.62) + bodies.length * 0.46 + (close.length ? 0.5 : 0));
  return { heads, bodies, close, stacked, ratio, ratioText: ratio >= 1 ? ratio.toFixed(1).replace(/\.0$/, "") + ":1" : "1:" + (1 / ratio).toFixed(1).replace(/\.0$/, "") };
}
// The slots in the order the model gets them: <Picture 1> is the first filled one.
const filledSlots = (list = slots()) => list.map((id, i) => (id ? i : -1)).filter((i) => i >= 0);
// The sheet's text, as the model reads it.
function sheetText(sh = state.prunaSheet, list = state.prunaSheetSlots, notes = state.drafts.prunaSheet) {
  if (sheetOn()) notes = state.prompt;
  const order = filledSlots(list), L = sheetLayout(sh.views);
  const views = Object.keys(SHEET_VIEWS).filter((v) => sh.views.includes(v)).map((v) => SHEET_VIEWS[v][2]);
  if (!views.length) return "";
  const helps = order.filter((i) => i > 0).map((i) => `Use <Picture ${order.indexOf(i) + 1}> for the ${sh.roles[i] === "detail" ? "details" : sh.roles[i]}.`);
  // It starts "Edit <Picture 1>", as P-H3-Image expects when it gets pictures, so the model doesn't add it again.
  return `Edit <Picture 1>: make a character reference sheet of this character, on a plain ${sh.bg} background with clear gaps between the views. `
    + `From left to right: ${views.join("; ")}. `
    + (L.stacked ? "The head views are stacked in two rows on the left. " : "")
    + "Same identity, face, hair, outfit, colours and proportions in every view; full figures from the top of the head to the soles with space above and below. No text, labels or borders."
    + (helps.length ? " " + helps.join(" ") : "") + ((notes || "").trim() ? " " + notes.trim().replace(/[^.!?]$/, "$&.") : "");
}
/* The Models page: model_catalog.json's groups, installed or not, downloaded in the background on the pod. */
const MODELS = { data: null, timer: null, open: false, focus: null, wasBusy: false, filter: "all" };
const TAB_NAMES = { video: "Video · MiniMax H3", image: "Image · MOSS-Image", klein: "Klein 9b", pruna: "Pruna · P-H3-Image" };
const bytesText = (b) => (!b ? "–" : b >= 1e9 ? (b / 1e9).toFixed(b >= 1e10 ? 0 : 1) + " GB" : b >= 1e6 ? Math.round(b / 1e6) + " MB" : Math.round(b / 1e3) + " kB");
const timeText = (s) => (s >= 3600 ? Math.floor(s / 3600) + " h " + Math.round((s % 3600) / 60) + " min" : s >= 60 ? Math.round(s / 60) + " min" : Math.max(1, Math.round(s)) + " s");
function openModels(tab) {
  MODELS.open = true; MODELS.focus = tab || null;
  $("models").hidden = false;
  pollModels(true);
}
function closeModels() { MODELS.open = false; $("models").hidden = true; }
async function pollModels(scroll) {
  clearTimeout(MODELS.timer);
  try { MODELS.data = await api("GET", "models"); } catch (error) { if (MODELS.open) $("modelsBody").innerHTML = `<p class="merr">${esc(error.message)}</p>`; return; }
  const busy = !!MODELS.data.job || MODELS.data.queue.length > 0;
  if (MODELS.wasBusy && !busy) load(false);   // something finished: what is installed changed
  MODELS.wasBusy = busy;
  if (MODELS.open) renderModels(scroll);
  if (MODELS.open || busy) MODELS.timer = setTimeout(() => pollModels(false), busy ? 1000 : 4000);
}
async function modelAction(id, action) {
  try {
    MODELS.data = await api("POST", `models/${id}/${action}`);
    if (action === "delete") { say(MODELS.data.kept.length ? "Deleted. Kept " + MODELS.data.kept.length + " file(s) another model needs." : "Deleted."); load(false); }
    if (action === "download") say("Downloading in the background. You can keep working.");
    MODELS.wasBusy = !!MODELS.data.job || MODELS.data.queue.length > 0;
    renderModels(false);
    pollModels(false);
  } catch (error) { say(error.message); }
}
// The Hugging Face token gated models (Klein) need: the pod's HF_TOKEN, or one pasted here. Drawn again only when
// it changes, so the field isn't cleared while you type.
function renderToken(t) {
  const key = JSON.stringify(t), box = $("modelsToken");
  if (box.dataset.key === key && !MODELS.tokenBusy) return;
  if (box.contains(document.activeElement) && box.dataset.key === key) return;
  box.dataset.key = key;
  if (t.from === "pod") {
    box.innerHTML = `<span class="token ok">Hugging Face token: the pod's HF_TOKEN</span>`;
  } else if (t.from) {
    box.innerHTML = `<span class="token ok">Hugging Face token${t.user ? " of " + esc(t.user) : ""} · ${t.from === "saved" ? "remembered on this pod" : "until ComfyUI restarts"}</span>
      <button class="link" data-act="tokenForget">Remove</button>`;
  } else {
    box.innerHTML = `<label class="mtoken-field"><span>Hugging Face token</span>
        <input id="hfToken" type="password" autocomplete="off" spellcheck="false" placeholder="hf_…" aria-describedby="hfTokenHint"></label>
      <label class="mtoken-keep"><input type="checkbox" id="hfRemember"> Remember on this pod</label>
      <button class="btn" data-act="tokenSave" id="hfTokenGo">Use token</button>
      <span class="hint" id="hfTokenHint">Gated models (Klein) need one: make a read token at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens</a> and accept the model's license there. Without Remember, it's kept until ComfyUI restarts.</span>`;
    $("hfToken").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); ACT.tokenSave(); } });
  }
}

// Add a video model from a Hugging Face link: a form of its own, so the page's redraws don't clear it.
function renderAddModel(add) {
  if (!add || $("mAddSampler").options.length) return;
  $("mAddSampler").innerHTML = add.samplers.map((x) => `<option>${esc(x)}</option>`).join("");
  $("mAddScheduler").innerHTML = add.schedulers.map((x) => `<option>${esc(x)}</option>`).join("");
}
$("mAddForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("mAddGo").disabled = true; $("mAddGo").textContent = "Checking the link…"; $("mAddErr").textContent = "";
  try {
    const r = await api("POST", "models/add", { url: $("mAddUrl").value, name: $("mAddName").value, steps: Number($("mAddSteps").value),
      sampler: $("mAddSampler").value, scheduler: $("mAddScheduler").value });
    MODELS.data = r; MODELS.wasBusy = true;
    $("mAddUrl").value = ""; $("mAddName").value = "";
    $("mAddForm").hidden = true; $("mAddOpen").hidden = false; $("mAddOpen").setAttribute("aria-expanded", "false");
    say("Added. Downloading in the background: it shows up among the video models when it's in.");
    MODELS.filter = "all"; renderModels(false); pollModels(false);
    document.getElementById("mg-" + r.added)?.scrollIntoView({ block: "center" });
  } catch (error) { $("mAddErr").textContent = error.message; }
  $("mAddGo").disabled = false; $("mAddGo").textContent = "Add and download";
});

function renderModels(scroll) {
  const d = MODELS.data;
  if (!d) return;
  // The models folder's disk: a bar of what is used, the downloads waiting shown on top of it, and the free space.
  const used = d.disk.total - d.disk.free, pct = (b) => Math.min(100, (b / d.disk.total) * 100);
  const pending = d.groups.filter((g) => g.state === "downloading" || g.state === "queued").reduce((n, g) => n + g.missingBytes, 0);
  const tight = d.disk.free - pending < 0.1 * d.disk.total;
  $("modelsDisk").innerHTML = `<span class="disk ${tight ? "tight" : ""}" title="${bytesText(used)} used of ${bytesText(d.disk.total)}">
      <span class="disk-bar"><span class="disk-used" style="width:${pct(used).toFixed(1)}%"></span><span class="disk-next" style="left:${pct(used).toFixed(1)}%;width:${pct(pending).toFixed(1)}%"></span></span>
      <span class="disk-text"><b>${bytesText(d.disk.free)}</b> free of ${bytesText(d.disk.total)}${pending ? ` · ${bytesText(pending)} on the way` : ""}</span></span>`;
  renderToken(d.token || {});
  renderAddModel(d.add);
  const byId = Object.fromEntries(d.groups.map((g) => [g.id, g]));
  const card = (g) => {
    const job = d.job && d.job.group === g.id ? d.job : null;
    const label = { installed: "Installed", partly: "Partly installed", missing: "Not installed", downloading: "Downloading", queued: "Waiting" }[g.state];
    const needs = (g.needs || []).map((n) => n.split("|").map((x) => byId[x]).filter(Boolean))
      .filter((alts) => alts.length && !alts.some((x) => x.state === "installed")).map((alts) => alts.map((x) => x.name).join(" or "));
    let prog = "";
    if (job) {
      const pct = job.total ? Math.min(100, (job.done / job.total) * 100) : 0, left = job.total && job.speed ? (job.total - job.done) / job.speed : 0;
      prog = `<div class="mprog"><span class="track"><span class="fill" style="width:${pct.toFixed(1)}%"></span></span>
        <span>File ${job.index} of ${job.count}: ${esc(job.file || "…")} · ${bytesText(job.done)}${job.total ? " of " + bytesText(job.total) : ""}${job.speed ? " · " + bytesText(job.speed) + "/s" : ""}${left ? " · " + timeText(left) + " left" : ""}</span></div>`;
    }
    const size = g.state === "installed" ? bytesText(g.bytes) : g.missingBytes ? bytesText(g.missingBytes) + (g.state === "partly" ? " to go" : "") + (g.unknownSize ? "+" : "") : g.unknownSize ? "size …" : "";
    return `<div class="mcard m-${g.state}" id="mg-${g.id}">
      <h3>${esc(g.name)} <span class="mstate ${g.state}">${label}</span>${g.gated && !(d.token && d.token.from) ? '<span class="mstate">Needs a Hugging Face token</span>' : ""}</h3>
      <p>${esc(g.about || "")}${g.url ? ` <a class="mlink" href="${esc(g.url)}" target="_blank" rel="noopener">On Hugging Face</a>` : ""}</p>
      ${needs.length ? `<span class="mneeds">Also needs: ${needs.map(esc).join(", ")}</span>` : ""}
      <div class="mside"><span class="msize">${size}</span><div class="mbtns">
        ${g.state === "downloading" || g.state === "queued" ? `<button class="btn" data-act="modelCancel" data-id="${g.id}">Cancel</button>`
          : g.custom && g.state === "installed" ? `<button class="btn danger" data-act="modelRemove" data-id="${g.id}">Remove</button>`
          : g.custom ? `<button class="btn" data-act="modelRemove" data-id="${g.id}">Remove</button><button class="btn primary" data-act="modelGet" data-id="${g.id}">${g.state === "partly" ? "Download the rest" : "Download"}</button>`
          : g.state === "installed" ? `<button class="btn danger" data-act="modelDelete" data-id="${g.id}">Delete</button>`
          : `${g.state === "partly" ? `<button class="btn danger" data-act="modelDelete" data-id="${g.id}">Delete</button>` : ""}<button class="btn primary" data-act="modelGet" data-id="${g.id}" ${g.missingBytes > d.disk.free ? "disabled title=\"Not enough free disk space\"" : ""}>${g.state === "partly" ? "Download the rest" : "Download"}</button>`}
      </div></div>
      ${prog}
      ${g.error ? `<span class="merr">${esc(g.error)}</span>` : ""}
      <details class="mfiles"><summary>${g.files.length} file${g.files.length === 1 ? "" : "s"}${g.env ? " · template switch " + esc(g.env) : ""}</summary><ul>${g.files.map((f) =>
        `<li class="${f.installed ? "ok" : ""}">${f.installed ? "✓" : "·"} ${esc(f.folder)}/${esc(f.name)} <span>${bytesText(f.size)}</span></li>`).join("")}</ul></details>
    </div>`;
  };
  const tabs = ["video", "image", "klein", "pruna"];
  // All / Installed / Not installed (partly installed, downloading and waiting count as not installed).
  const shows = (g) => MODELS.filter === "all" || (MODELS.filter === "installed") === (g.state === "installed");
  const count = (f) => d.groups.filter((g) => f === "all" || (f === "installed") === (g.state === "installed")).length;
  $("modelsFilter").innerHTML = [["all", "All"], ["installed", "Installed"], ["missing", "Not installed"]].map(([id, label]) =>
    `<button class="${MODELS.filter === id ? "on" : ""}" data-act="modelsFilter" data-id="${id}" aria-pressed="${MODELS.filter === id}">${label} · ${count(id)}</button>`).join("");
  const open = document.querySelectorAll(".mfiles[open]");
  const keep = new Set([...open].map((el) => el.closest(".mcard").id));
  const top = $("modelsBody").scrollTop;
  const shown = tabs.map((t) => [t, d.groups.filter((g) => g.tab === t && shows(g))]).filter(([, list]) => list.length);
  $("modelsBody").innerHTML = (shown.length ? shown.map(([t, list]) => `<h4 class="models-tab" id="mt-${t}">${TAB_NAMES[t]}</h4>${list.map(card).join("")}`).join("")
    : `<p class="models-note">${MODELS.filter === "installed" ? "Nothing is installed yet." : "Everything is installed."}</p>`)
    + `<p class="models-note">Downloads run on the pod, one file at a time, and carry on while you work or if you close this page. A model shows up in its tab when its files are in. On RunPod, the template's switches (shown with each model) download the same files when the pod starts.</p>`;
  keep.forEach((id) => document.querySelector("#" + id + " .mfiles")?.setAttribute("open", ""));
  if (scroll && MODELS.focus) $("mt-" + MODELS.focus)?.scrollIntoView({ block: "start" });
  else $("modelsBody").scrollTop = top;
}

// The head painter: paint over the body picture where it may change (the head and neck). The strokes are
// kept as a white-on-clear PNG on the server, for Only the head.
const painter = { erase: false, size: 40, drawing: false, last: null };
function openPainter() {
  const body = slotRef(0, state.kleinSwapSlots);
  if (!body) { say("Add the body picture first."); return; }
  const img = new Image();
  img.onload = () => {
    const k = Math.min(1, 2048 / Math.max(img.naturalWidth, img.naturalHeight));
    const c = $("paintCanvas");
    c.width = Math.round(img.naturalWidth * k); c.height = Math.round(img.naturalHeight * k);
    $("paintImg").src = body.img;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    const old = maskFor(body.id);
    if (old && old.url) {
      const m = new Image();
      m.onload = () => { ctx.drawImage(m, 0, 0, c.width, c.height); ctx.globalCompositeOperation = "source-in"; ctx.fillStyle = "#ff4b3e"; ctx.fillRect(0, 0, c.width, c.height); ctx.globalCompositeOperation = "source-over"; };
      m.src = old.url;
    }
    painter.erase = false; $("paintErase").classList.remove("on");
    $("paintSize").value = painter.size;
    $("painter").hidden = false;
  };
  img.onerror = () => say("Couldn't load the body picture.");
  img.src = body.img;
}
function closePainter() { $("painter").hidden = true; }
async function finishPainter() {
  const c = $("paintCanvas"), body = state.kleinSwapSlots[0];
  const data = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  let any = false;
  for (let i = 3; i < data.length; i += 4 * 16) if (data[i] > 20) { any = true; break; }
  if (!any) { state.kleinSwapMask = null; state.kleinSettings.headOnly = false; state.kleinSettings.refine = false; closePainter(); render(); saveUi(); say("No head painted: the whole picture is made again."); return; }
  // White where it was painted, clear elsewhere.
  const out = document.createElement("canvas");
  out.width = c.width; out.height = c.height;
  const o = out.getContext("2d");
  o.drawImage(c, 0, 0); o.globalCompositeOperation = "source-in"; o.fillStyle = "#fff"; o.fillRect(0, 0, out.width, out.height);
  try {
    const { mask } = await api("POST", "masks", { data: out.toDataURL("image/png") });
    state.kleinSwapMask = { image: body, mask, url: out.toDataURL("image/png") };
    state.kleinSettings.headOnly = true;
    closePainter(); render(); saveUi();
    say("Head painted: only that part of the body picture changes.");
  } catch (error) { say(error.message); }
}
(() => {
  const at = (e) => { const c = $("paintCanvas"), r = c.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * c.width, ((e.clientY - r.top) / r.height) * c.height, c.width / r.width]; };
  const stroke = (from, to) => {
    const ctx = $("paintCanvas").getContext("2d");
    ctx.globalCompositeOperation = painter.erase ? "destination-out" : "source-over";
    ctx.strokeStyle = "#ff4b3e"; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.lineWidth = painter.size * to[2];
    ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke();
    ctx.globalCompositeOperation = "source-over";
  };
  document.addEventListener("pointerdown", (e) => {
    if (e.target.id !== "paintCanvas") return;
    e.preventDefault(); painter.drawing = true; painter.last = at(e); stroke(painter.last, painter.last);
    e.target.setPointerCapture(e.pointerId);
  });
  document.addEventListener("pointermove", (e) => {
    const c = $("paintCanvas");
    if (e.target === c || painter.drawing) {
      const r = c.getBoundingClientRect(), dot = $("paintDot");
      dot.style.width = dot.style.height = painter.size + "px";
      dot.style.left = e.clientX - r.left + "px"; dot.style.top = e.clientY - r.top + "px";
    }
    if (!painter.drawing) return;
    const p = at(e); stroke(painter.last, p); painter.last = p;
  });
  document.addEventListener("pointerup", () => { painter.drawing = false; });
  document.addEventListener("input", (e) => { if (e.target.id === "paintSize") painter.size = Number(e.target.value); });
})();

// A face swap's before/after: the body picture over the result, cut at a slider you drag.
const compareBody = (c) => (c.swap && c.status === "done" && c.slots && c.slots[0] && state.refs.find((r) => r.id === c.slots[0].id)) || null;
function compareHTML(c) {
  if (!compareBody(c)) return "";
  return state.compare
    ? `<span class="compare-tag before">Before</span><span class="compare-tag after">After</span><button class="compare-btn on" data-act="compareToggle">Close compare</button>`
    : `<button class="compare-btn" data-act="compareToggle" title="Compare with the body picture">Before / after</button>`;
}
// The body picture over the result, shown left of the cut; drag across the picture to move it.
function compareImg(c) {
  const body = compareBody(c);
  return body && state.compare ? `<span class="compare" id="compare" style="--cut:50%"><img src="${body.img}" alt="Before" draggable="false"><span class="compare-line"></span></span>` : "";
}
document.addEventListener("pointerdown", (e) => {
  const box = e.target.closest("#compare");
  if (!box) return;
  e.preventDefault();
  const move = (ev) => { const r = box.getBoundingClientRect(); box.style.setProperty("--cut", Math.max(0, Math.min(100, ((ev.clientX - r.left) / r.width) * 100)) + "%"); };
  move(e);
  const up = () => { document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", up); };
  document.addEventListener("pointermove", move); document.addEventListener("pointerup", up);
});

// Klein's face swap prompt (from the server, the faceswap detailer workflow's), with the extra faces and notes.
function swapText(notes = state.prompt) {
  const faces = state.kleinSwapSlots.filter(Boolean).length;
  let text = OPT.klein.swapPrompt || "head_swap: Use image 1 as the base image …";
  if (faces > 2) text += "\n" + ({ 3: "Image 3 shows", 4: "Images 3 and 4 show" }[faces] || `Images 3 to ${faces} show`) + " the same person as image 2 from other angles: use them for the identity too.";
  return text + (notes.trim() ? "\n" + notes.trim() : "");
}
// The head mask for this body picture, if one is painted.
const maskFor = (id) => (id && state.kleinSwapMask && state.kleinSwapMask.image === id ? state.kleinSwapMask : null);
const headOnly = () => swapOn() && state.kleinSettings.headOnly && !!maskFor(state.kleinSwapSlots[0]);

// What P-H3-Image (or with Enhance, its prompt upsampler) gets, as the server sends it.
const PH3_RATIOS = { "1:1": 1, "4:3": 4 / 3, "3:2": 1.5, "16:9": 16 / 9, "21:9": 21 / 9, "3:1": 3, "4:1": 4, "3:4": 3 / 4, "2:3": 2 / 3, "9:16": 9 / 16 };
function prunaSent() {
  const h3 = h3img(modelOf(state.prunaSettings.model)), [w, h] = prunaSize(), enhance = state.prunaSettings.enhance && !h3;
  const order = filledSlots();
  let text = sheetOn() ? sheetText()
    : state.prompt.trim().replace(/\bpicture\s?([1-9])\b/gi, (m, n) => { const k = order.indexOf(n - 1); return k < 0 ? m : `<Picture ${k + 1}>`; });
  if (!text) return "";
  if (enhance) {
    const ar = Object.keys(PH3_RATIOS).reduce((a, b) => (Math.abs(Math.log(PH3_RATIOS[b] / (w / h))) < Math.abs(Math.log(PH3_RATIOS[a] / (w / h))) ? b : a));
    return `Aspect ratio: ${ar}\nRequest: ${text}`;
  }
  return order.length && !h3 && !/^\s*(Edit <Picture \d+>|subject_definitions:)/.test(text) ? "Edit <Picture 1>. " + text : text;
}

// The slots the prompt names that have nothing in them.
const emptyNamed = (text = state.prompt) => (sheetOn() || swapOn() || !isSlotTab() ? []
  : [...new Set([...text.matchAll(slotRe())].map((m) => Number(m[1])))].filter((n) => !slots()[n - 1]).sort());
function renderSlots() {
  $("refs").hidden = false;
  document.querySelector(".composer").classList.remove("no-refs");
  if (editOn()) { renderEditSlot(); return; }
  const list = slots(), filled = list.filter(Boolean).length, empty = emptyNamed(), word = isPruna() ? "picture" : "image";
  $("hint").textContent = empty.length
    ? `The prompt names ${empty.map((n) => word + " " + n).join(" and ")}, but ${empty.length > 1 ? "those slots are" : "that slot is"} empty`
    : swapOn() ? (list[0] && list[1] ? (list.filter(Boolean).length > 2 ? "The face goes onto the body; the other face pictures help with who it is." : "The face goes onto the body. More pictures of the same face help; notes are optional.")
      : list[0] ? "Now the face: the person whose head goes onto the body" : "First the body and scene: the picture that keeps its pose, light and background")
    : sheetOn() ? (list[0] ? `${state.prunaSheet.views.length} view${state.prunaSheet.views.length === 1 ? "" : "s"} · the extra pictures help with the face, outfit or details` : "Add the character to draw")
    : isPruna() ? (filled ? `${filled} of 9 pictures · picture 1 is edited, the others are references` : "No pictures: it makes one from the prompt alone")
    : filled ? `${filled} of 4 pictures · drag one onto another slot to swap them` : "No pictures: Klein makes one from the prompt alone";
  $("hint").classList.toggle("warn", empty.length > 0);
  // Pruna's Image shows every filled slot and one more (at least four, at most nine).
  const shown = isPruna() && !sheetOn() ? Math.min(9, Math.max(4, list.map(Boolean).lastIndexOf(true) + 2))
    // Face swap: body and face, then one more face slot at a time (up to five pictures).
    : swapOn() ? Math.min(5, Math.max(2, list.map(Boolean).lastIndexOf(true) + 2)) : list.length;
  $("refs").innerHTML = list.slice(0, shown).map((id, i) => {
    const r = slotRef(i), name = slotWord(i);
    const role = sheetOn() && i > 0 ? `<select class="slot-role" data-change="sheetRole" data-id="${i}" aria-label="What picture ${i + 1} shows">${Object.entries(SHEET_ROLES).map(([k, v]) => `<option value="${k}" ${state.prunaSheet.roles[i] === k ? "selected" : ""}>${v}</option>`).join("")}</select>` : "";
    if (!r) return `<div class="ref slot ${sheetOn() && !i ? "first" : ""}">
        <button class="slot-empty" data-act="pickSlot" data-id="${i}" data-slot="${i}" aria-label="Add ${name}" title="Add a picture here: click, or drag one in from the grid, Add image or your computer">${ICON.plus}${name}</button>
        <span class="slot-res"></span>${role}</div>`;
    refSize(r);
    return `<div class="ref slot used">
        <span class="ref-pic slot-pic" style="background:${r.tone}" draggable="true" data-drag-slot="${i}" data-slot="${i}" data-act="showSlot" data-id="${i}" title="${name}: @${esc(r.name)}. Click to open Add image and replace it.${sheetOn() ? "" : ` Drag it into the prompt to put “${name}” at the cursor, or onto another slot to swap them.`}">
          <img src="${r.img}" alt="" draggable="false"><span class="slot-name">${name}</span>
          <button class="ref-x" data-act="clearSlot" data-id="${i}" aria-label="Empty ${name}">${ICON.x}</button>
        </span>
        <span class="slot-res">${r.w ? r.w + "×" + r.h : ""}</span>${role}${swapOn() && i === 0 ? `<button class="slot-paint ${maskFor(id) ? "on" : ""}" data-act="paintMask" title="Paint the head: only that part of the body picture changes">${maskFor(id) ? "Head painted" : "Paint head"}</button>` : ""}</div>`;
  }).join("") + `<button class="add-ref" id="addRef" data-act="openLibrary" aria-expanded="${state.library.open}">${ICON.plus}Add image</button>`;
}
// The Image tab's Edit slot: the picture to edit, with its size under it.
function renderEditSlot() {
  const r = slotRef(0);
  $("hint").classList.remove("warn");
  $("hint").textContent = r ? "Edit: describe only the change, e.g. “Change only the jacket to a black wool coat. Keep everything else unchanged.”"
    : "Edit: pick the picture to edit";
  if (!r) {
    $("refs").innerHTML = `<div class="ref slot">
        <button class="slot-empty edit-slot" data-act="pickSlot" data-id="0" data-slot="0" aria-label="Add the picture to edit" title="Add the picture to edit: click, or drag one in from the grid, Add image or your computer">${ICON.plus}Picture<br>to edit</button>
        <span class="slot-res"></span></div>`;
    return;
  }
  refSize(r);
  $("refs").innerHTML = `<div class="ref slot used">
      <span class="ref-pic slot-pic" style="background:${r.tone}" draggable="true" data-drag-slot="0" data-slot="0" data-act="showSlot" data-id="0" title="The picture to edit: @${esc(r.name)}. Click to open Add image and pick another, or drop one here.">
        <img src="${r.img}" alt="" draggable="false"><span class="slot-name">Edit</span>
        <button class="ref-x" data-act="clearSlot" data-id="0" aria-label="Take out the picture to edit">${ICON.x}</button>
      </span>
      <span class="slot-res">${r.w ? r.w + "×" + r.h : ""}</span></div>`;
}

// Put a picture (an Add image library entry) in a slot, or in the first empty one. Returns the slot, or -1.
function putInSlot(ref, slot = -1) {
  const list = slots();
  if (slot < 0) slot = list.indexOf(null);
  if (slot < 0 && editOn()) slot = 0;   // Edit has one slot: a new picture replaces it
  if (slot < 0) { say("All the slots are full. Drop it on one to replace that picture."); return -1; }
  list[slot] = ref.id;
  render();
  say(isSlotTab() ? `@${ref.name} is ${sheetOn() || swapOn() ? "the " : ""}${slotWord(slot)}.` : `Editing @${ref.name}.`);
  return slot;
}

function renderPrompt() {
  $("mirror").innerHTML = state.prompt
    ? promptHTML(state.prompt) + "​"
    : `<span class="ph">${editOn() ? "Describe only the change, for example: Change only the pink blazer to a plain black wool coat. Keep the faces, poses and background unchanged."
      : isImage() ? "Describe an image, for example: a neon sign that says \"OPEN ALL NIGHT\" in the rain."
      : isKlein() ? "Describe the edit. Call the pictures image 1, image 2, image 3 and image 4, for example: put the bag from image 2 in the hand of the woman in image 1."
      : sheetOn() ? "Optional notes for the sheet, for example: keep the scarf, no weapon."
      : swapOn() ? "Optional notes, added after the face swap prompt, for example: keep the glasses from the face."
      : isPruna() ? "Describe a picture, or the edit to picture 1. Call the pictures picture 1, picture 2 …, for example: put the lamp from picture 2 on the table in picture 1."
      : "Describe a shot. With pictures, write it as MiniMax does: &lt;Subject 1&gt; is … in &lt;Picture 1&gt;. (type @ to pick a picture)"}</span>`;
  if ($("prompt").value !== state.prompt) $("prompt").value = state.prompt;
  fitPrompt();
  renderMention();
}

/* @-mentions */
function mentionQuery() {
  if (state.mentionClosed || imageTab()) return null;
  const m = /(^|\s)@([A-Za-z0-9_]*)$/.exec(state.prompt.slice(0, state.caret));
  return m ? m[2] : null;
}

function mentionMatches() {
  const q = mentionQuery();
  if (q === null) return [];
  // Video: the prompt bar's pictures, as <Picture N>.
  if (state.tab === "video") {
    const ql = q.toLowerCase();
    return state.attached.map((id, i) => ({ ...state.refs.find((r) => r.id === id), picture: i }))
      .filter((r) => r.id && (!ql || String(r.picture + 1).startsWith(ql) || "picture".startsWith(ql) || r.name.toLowerCase().startsWith(ql)));
  }
  return state.refs.filter((r) => r.name.toLowerCase().startsWith(q.toLowerCase()));
}

function renderMention() {
  const q = mentionQuery();
  const box = $("mention");
  if (q === null) { box.hidden = true; return; }
  const list = mentionMatches();
  const idx = Math.min(state.mentionIndex, Math.max(list.length - 1, 0));
  const used = new Set(mentionsIn(state.prompt));
  box.hidden = false;
  if (state.tab === "video") {
    box.innerHTML = `<span class="mention-title">${list.length ? "Pictures in the prompt bar" : state.attached.length ? "No picture matches" : "No pictures yet: Add image puts one in the prompt bar."}</span>`
      + list.map((r, i) => `
      <button class="mention-item ${i === idx ? "on" : ""}" data-act="pickMention" data-id="p${r.picture}" role="option" aria-selected="${i === idx}">
        <span class="mention-thumb" style="background:${r.tone}"><img src="${r.img}" alt=""></span>
        <span>&lt;Picture ${r.picture + 1}&gt;<span class="mention-sub">${esc(r.name)}</span></span>
      </button>`).join("") + `<span class="mention-help">↑ ↓ to pick · Enter to insert · Esc to close</span>`;
    return;
  }
  box.innerHTML = `<span class="mention-title">${list.length ? (q ? `Images starting with "${esc(q)}"` : "Your images") : state.refs.length ? `No image called @${esc(q)}` : "No images yet. Use Add image to upload one."}</span>`
    + list.map((r, i) => `
      <button class="mention-item ${i === idx ? "on" : ""}" data-act="pickMention" data-id="${r.id}" role="option" aria-selected="${i === idx}">
        <span class="mention-thumb" style="background:${r.tone}"><img src="${r.img}" alt=""></span>
        <span>@${esc(r.name)}<span class="mention-sub">${used.has(r.name) ? "Already in the prompt" : "Image"}</span></span>
      </button>`).join("")
    + `<span class="mention-help">↑ ↓ to pick · Enter to insert · Esc to close</span>`;
}

// Open Add image (if it isn't open) with this picture picked, from wherever it came.
function showInLibrary(r) {
  if (!r) return;
  state.library.open = true;
  state.settingsOpen = false; state.loraPicker = false;
  if (r.from_clip && clipById(r.from_clip)) { state.library.source = "generated"; state.library.selectedGen = r.from_clip; }
  else { state.library.source = "uploads"; state.library.selected = r.id; }
  render();
  document.querySelector("#libList .lib-item.on")?.scrollIntoView({ block: "nearest" });
}

// Put a word (an @name, or Klein's "image 2") in the prompt where the cursor was, with spaces around it,
// and put the cursor after it. at: a position in the prompt instead (where something was dropped).
function insertWord(word, at = null) {
  const pos = wordEdge(state.prompt, Math.min(at ?? state.caret ?? state.prompt.length, state.prompt.length));
  const before = state.prompt.slice(0, pos), after = state.prompt.slice(pos);
  const insert = (before && !/\s$/.test(before) ? " " : "") + word + (/^\s/.test(after) ? "" : " ");
  state.prompt = before + insert + after;
  state.caret = before.length + insert.length;
  state.mentionClosed = true;
  // The text stays where it was scrolled to: setting the cursor would jump to it (often the end).
  const scrolled = ta.scrollTop;
  renderPrompt(); renderRefs();
  saveUi();
  ta.focus({ preventScroll: true });
  ta.setSelectionRange(state.caret, state.caret);
  ta.scrollTop = scrolled; $("mirror").scrollTop = scrolled;
}

function insertMention(ref) {
  const word = ref.picture != null ? `<Picture ${ref.picture + 1}>` : "@" + ref.name;
  if (ref.picture == null) attach(ref.id);
  const before = state.prompt.slice(0, state.caret).replace(/@([A-Za-z0-9_]*)$/, word + " ");
  const after = state.prompt.slice(state.caret).replace(/^[A-Za-z0-9_]*\s?/, "");
  state.prompt = before + after;
  state.caret = before.length;
  state.mentionIndex = 0;
  state.mentionClosed = true;
  renderPrompt();
  renderRefs();
  saveUi();
  const ta = $("prompt");
  ta.focus();
  ta.setSelectionRange(state.caret, state.caret);
}

/* Image library: a pop-up above the prompt bar */
// The library has two sources: your uploads, and the finished images from the Image tab.
// An image from the Image tab becomes an @image (a library entry) the first time it is used.
const libSources = () => [
  ["uploads", "Uploads", ICON.upload, uploadItems().length],
  ["generated", "Generated images", ICON.image, state.clips.filter((c) => kindOf(c) === "image" && c.status === "done").length],
];
const isGen = () => state.library.source === "generated";
const refForClip = (clipId) => state.refs.find((r) => r.from_clip === clipId);

function uploadItems() {
  const L = state.library;
  const q = L.query.trim().toLowerCase();
  const items = state.refs.filter((r) => !r.from_clip && (!q || r.name.toLowerCase().includes(q)));
  const order = {
    newest: (a, b) => b.added - a.added,
    oldest: (a, b) => a.added - b.added,
    name: (a, b) => a.name.localeCompare(b.name),
  }[L.sort];
  return items.sort(order);
}

function generatedItems() {
  const L = state.library;
  const q = L.query.trim().toLowerCase();
  const items = state.clips.filter((c) => kindOf(c) === "image" && c.status === "done" && (!q || c.prompt.toLowerCase().includes(q)
    || (refForClip(c.id)?.name || "").includes(q)));
  const order = {
    newest: (a, b) => (b.created || 0) - (a.created || 0),
    oldest: (a, b) => (a.created || 0) - (b.created || 0),
    name: (a, b) => a.prompt.localeCompare(b.prompt),
  }[L.sort];
  return items.sort(order);
}

// The ids shown in the list now, in order (for the arrow keys).
const libraryIds = () => (isGen() ? generatedItems() : uploadItems()).map((x) => x.id);
const libSelected = () => (isGen() ? state.library.selectedGen : state.library.selected);

// Where pop-ups above the prompt bar may reach up to: a little under the top bar.
const belowTop = () => (document.querySelector(".top")?.getBoundingClientRect().bottom || 0) + 12;

function renderLibrary() {
  const box = $("library");
  // Closing, its lists go back to the top (before it hides: a hidden list can't scroll), so it opens at the top.
  if (!state.library.open) {
    if (!box.hidden) box.querySelectorAll("#libList, #libNav, #libPreview").forEach((el) => { el.scrollTop = 0; });
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.style.maxHeight = Math.max(320, document.querySelector(".composer").getBoundingClientRect().top - 12 - belowTop()) + "px";
  if (!box.firstChild) {
    // The frame is drawn once, so the search field keeps focus while you type.
    box.innerHTML = `
      <div class="lib-top">
        <label class="lib-search">${ICON.search}<input id="libQuery" type="search" placeholder="Search images" aria-label="Search images" data-input="libQuery" aria-controls="libList"></label>
        <select class="select lib-sort" aria-label="Sort images" data-change="libSort">
          <option value="newest">Newest</option><option value="oldest">Oldest</option><option value="name">Name A–Z</option>
        </select>
      </div>
      <div class="lib-body">
        <nav class="lib-nav" id="libNav" aria-label="Where the images come from"></nav>
        <div class="lib-list" id="libList" role="listbox" aria-label="Your images"></div>
        <div class="lib-preview" id="libPreview"></div>
      </div>`;
  }
  box.querySelector(".lib-sort").value = state.library.sort;
  renderLibNav();
  renderLibList();
  renderLibPreview();
}

function renderLibNav() {
  $("libNav").innerHTML = libSources().map(([id, label, icon, count]) => `
    <button class="lib-src ${state.library.source === id ? "on" : ""}" data-act="libSource" data-id="${id}" aria-pressed="${state.library.source === id}">
      ${icon}<span>${label}</span><span class="lib-count">${count}</span>
    </button>`).join("");
}

function renderLibList() {
  if (isGen()) { renderGenList(); return; }
  const items = uploadItems();
  $("libList").innerHTML = `
    <label class="lib-upload">${ICON.upload}Upload image
      <input type="file" accept="image/*" multiple data-change="addImages">
    </label>`
    + (items.length ? items.map((r) => `
      <div class="lib-item ${r.id === state.library.selected ? "on" : ""}" data-act="libSelect" data-id="${r.id}" role="option" tabindex="${r.id === state.library.selected ? 0 : -1}" aria-selected="${r.id === state.library.selected}" draggable="true" data-drag-ref="${r.id}" title="Drag into the prompt">
        <span class="lib-thumb" style="background:${r.tone}"><img src="${r.img}" alt="" draggable="false"></span>
        <span class="lib-text"><span class="lib-name">@${esc(r.name)}</span><span class="lib-sub">${state.attached.includes(r.id) ? "Image · in the prompt bar" : "Image"}</span></span>
      </div>`).join("")
    : `<span class="lib-none">${state.refs.some((r) => !r.from_clip) ? `No images match "${esc(state.library.query)}".` : "No uploads yet. Upload one, or drop images anywhere on the page."}</span>`);
}

function renderGenList() {
  const items = generatedItems();
  $("libList").innerHTML = items.length ? items.map((c) => {
    const ref = refForClip(c.id), on = c.id === state.library.selectedGen;
    return `
      <div class="lib-item ${on ? "on" : ""}" data-act="libSelect" data-id="${c.id}" role="option" tabindex="${on ? 0 : -1}" aria-selected="${on}" draggable="true" data-drag-gen="${c.id}" title="Drag into the prompt">
        <span class="lib-thumb" style="background:${toneOf(c.id)}"><img src="${clipUrl(c)}" alt="" draggable="false"></span>
        <span class="lib-text"><span class="lib-name">${ref ? "@" + esc(ref.name) : esc(c.prompt)}</span><span class="lib-sub">${ref && state.attached.includes(ref.id) ? "In the prompt bar" : esc(c.model.replace(/ \(.*\)$/, "")) + " · " + c.aspect}</span></span>
      </div>`;
  }).join("")
    : `<span class="lib-none">${state.clips.some((c) => kindOf(c) === "image" && c.status === "done") ? `No images match "${esc(state.library.query)}".`
      : `No generated images yet. Make some in the <button class="link inline" data-act="tab" data-id="image">Image tab</button>.`}</span>`;
}

function renderLibPreview() {
  if (isGen()) {
    const c = clipById(state.library.selectedGen);
    if (!c || kindOf(c) !== "image") { $("libPreview").innerHTML = `<span class="lib-none">Pick an image to see it here.</span>`; return; }
    const ref = refForClip(c.id), inBar = ref && state.attached.includes(ref.id);
    $("libPreview").innerHTML = `
      <div class="lib-big" draggable="true" data-drag-gen="${c.id}" data-act="zoomPic" title="Click to see it bigger. Drag into the prompt"><img src="${clipUrl(c)}" alt="${esc(c.prompt)}" draggable="false"></div>
      <p class="lib-caption">${esc(c.prompt)}</p>
      ${libButton(ref?.id, inBar)}`;
    return;
  }
  const r = state.refs.find((x) => x.id === state.library.selected);
  if (!r) { $("libPreview").innerHTML = `<span class="lib-none">Pick an image to see it here.</span>`; return; }
  const inBar = state.attached.includes(r.id);
  $("libPreview").innerHTML = `
    <div class="lib-big" draggable="true" data-drag-ref="${r.id}" data-act="zoomPic" title="Click to see it bigger. Drag into the prompt"><img src="${r.img}" alt="@${esc(r.name)}" draggable="false"></div>
    ${libButton(r.id, inBar)}`;
}
// The button under the library's big preview: into the prompt bar, or (Klein) into a slot.
function libButton(refId, inBar) {
  if (editOn()) {
    if (refId && state.imageEditSlot[0] === refId) return `<button class="btn lib-add" data-act="libToggle">Don't edit this picture</button>`;
    return `<button class="btn primary lib-add" data-act="libToggle">Edit this picture</button>`;
  }
  if (isSlotTab()) {
    const list = slots();
    const at = refId ? list.indexOf(refId) : -1;
    const to = state.library.slot != null ? state.library.slot : list.indexOf(null);
    if (at >= 0) return `<button class="btn lib-add" data-act="libToggle">Take out of ${sheetOn() || swapOn() ? "the " : ""}${slotWord(at)}</button>`;
    return `<button class="btn primary lib-add" data-act="libToggle" ${to < 0 ? "disabled" : ""}>${to < 0 ? "All the slots are full" : "Put in " + (sheetOn() || swapOn() ? "the " : "") + slotWord(to)}</button>`;
  }
  if (state.tab === "video") {
    const n = state.attached.indexOf(refId);
    if (n >= 0) return `<button class="btn lib-add" data-act="libToggle">Take out (it's Picture ${n + 1})</button>`;
    return `<button class="btn primary lib-add" data-act="libToggle" ${state.attached.length >= 9 ? "disabled" : ""}>${state.attached.length >= 9 ? "H3 takes at most 9 pictures" : "Add as Picture " + (state.attached.length + 1)}</button>`;
  }
  return `<button class="btn ${inBar ? "" : "primary"} lib-add" data-act="libToggle">${inBar ? "Remove from prompt" : "Add to prompt"}</button>`;
}

// Move the selection up or down the list, and keep it in view.
function libMove(delta) {
  const ids = libraryIds();
  if (!ids.length) return;
  const i = ids.indexOf(libSelected());
  const next = ids[i < 0 ? (delta > 0 ? 0 : ids.length - 1) : Math.max(0, Math.min(ids.length - 1, i + delta))];
  ACT.libSelect(next, null, null, true);
}

// The @image for a finished image from the Image tab, made the first time it is needed.
async function refFromClip(clipId) {
  const have = refForClip(clipId);
  if (have) return have;
  const { image } = await api("POST", "images/from-clip", { clip_id: clipId });
  const ref = fromImage(image);
  if (!state.refs.some((r) => r.id === ref.id)) state.refs.push(ref);
  return ref;
}

/* Settings: a pop-up that opens above its button */
function renderSettings() {
  const s = state.settings;
  $("pill").textContent = summary();
  $("pill").setAttribute("aria-expanded", state.settingsOpen);
  const box = $("settings");
  box.hidden = !state.settingsOpen;
  if (!state.settingsOpen) return;
  if (document.activeElement?.id === "valueInput") return;   // a value being typed in
  // Fit between the top bar and the button.
  box.style.maxHeight = Math.max(240, $("pill").getBoundingClientRect().top - 12 - belowTop()) + "px";

  const T = tabOpt();
  const model = modelOf(s.model);
  const minSteps = model.minSteps || 1, stepSize = model.stepSize || 1;
  const recPct = ((model.steps - minSteps) / (model.maxSteps - minSteps)) * 100;
  if (isKlein()) { box.innerHTML = kleinSettingsHTML(s, model, T, minSteps, stepSize, recPct); return; }
  if (isPruna()) { box.innerHTML = prunaSettingsHTML(s, model, T); return; }
  box.innerHTML = motionNote(model) + `
    <div class="field">
      <label class="label" for="model">Model</label>
      <select class="select" id="model" data-change="model">
        ${T.models.map((m) => `<option value="${esc(m.name)}" ${m.name === s.model ? "selected" : ""}>${esc(m.name)}</option>`).join("")}
      </select>
    </div>

    ${editOn() ? `<div class="field">
      <span class="label">Size</span>
      <span class="hint">${editSize() ? `The picture's shape: ${editSize().join("×")} at ${mpText(s.res)}.` : "The picture's shape, at the resolution below."}</span>
    </div>` : `<div class="field">
      <span class="label">Aspect ratio</span>
      <div class="aspects ${isImage() ? "five" : ""}">
        ${Object.keys(T.aspects).map((a) => {
          const [w, h] = T.aspects[a];
          const k = 24 / Math.max(w, h);
          return `<button class="${a === s.aspect ? "on" : ""}" data-act="aspect" data-id="${a}"><i style="width:${Math.round(w * k)}px;height:${Math.round(h * k)}px"></i>${a}</button>`;
        }).join("")}
      </div>
    </div>`}

    ${isImage() ? "" : `<div class="field">
      <div class="field-head"><label class="label" for="len">Length</label><button class="value value-edit" id="lenVal" data-act="editValue" data-id="len" title="Click to type a length">${s.len} s</button></div>
      <input id="len" type="range" min="${OPT.length.min}" max="${OPT.length.max}" step="1" value="${s.len}" data-input="len">
    </div>`}

    ${isImage() ? `<div class="field">
      <label class="label" for="res">Resolution</label>
      <select class="select" id="res" data-change="res">
        ${T.megapixels.map((mp) => `<option value="${mp}" ${mp === s.res ? "selected" : ""}>${mpText(mp)} (${sizeText(s.aspect, mp)})${(T.recommendedMP || []).includes(mp) ? " · recommended" : ""}</option>`).join("")}
      </select>
    </div>` : videoResHTML(s, T)}

    <div class="field">
      <div class="field-head"><label class="label" for="steps">Steps</label><button class="value value-edit ${s.steps === model.steps ? "rec" : ""}" id="stepsVal" data-act="editValue" data-id="steps" title="Click to type a number of steps">${stepsText(s.steps, model.steps)}</button></div>
      <div class="steps-wrap">
        <input id="steps" type="range" min="${minSteps}" max="${model.maxSteps}" step="${stepSize}" value="${s.steps}" data-input="steps" aria-describedby="stepsRec">
        <span class="rec-mark" style="left:calc(${recPct}% + ${(0.5 - recPct / 100) * 16}px)" aria-hidden="true"></span>
      </div>
      <div class="field-head steps-foot"><span class="hint" id="stepsRec">Recommended for ${esc(model.name)}: ${model.steps}${stepSize > 1 ? ` · takes ${minSteps}–${model.maxSteps} in steps of ${stepSize}` : ""}</span>
        <button class="link" data-act="recSteps" id="recBtn" style="visibility:${s.steps === model.steps ? "hidden" : "visible"}">Use ${model.steps}</button></div>
    </div>

    ${isImage() ? guidanceField(s, model) : ""}

    <div class="field">
      <span class="label">Seed</span>
      <div class="seed-row">
        <div class="seg">
          <button class="${s.seedMode === "random" ? "on" : ""}" data-act="seedMode" data-id="random">Random</button>
          <button class="${s.seedMode === "fixed" ? "on" : ""}" data-act="seedMode" data-id="fixed">Fixed</button>
        </div>
        ${s.seedMode === "fixed"
          ? `<input class="seed-input" inputmode="numeric" aria-label="Seed number" value="${s.seed}" data-change="seed">
             <button class="icon-btn" data-act="rollSeed" aria-label="New random seed">${ICON.dice}</button>`
          : `<span class="seed-note">A new seed every run, saved with each ${isImage() ? "image" : "clip"}.</span>`}
      </div>
    </div>

    ${state.tab === "video" ? realismField(s) : ""}
    ${lorasField(s)}
    ${state.tab === "video" ? refmodsField(s) : ""}

    ${samplingFields(s, model)}`;
}

// ---- RefMods (Video, with the ComfyUI-MiniMaxH3Mod pack) ----
// Saved references made from pictures (no training); a render carries up to eight, each with a strength.
const RM = { installed: false, files: [], jobs: [], loaded: false, drive: { state: "off" } };
let rmDriveTimer = null;
async function loadRefmods(sync) {
  try { Object.assign(RM, await api("GET", "refmods" + (sync ? "?sync=1" : "")), { loaded: true }); } catch { return; }
  // While Google Drive is being copied to or from, look again in a moment.
  clearTimeout(rmDriveTimer);
  if (RM.drive.state === "syncing") rmDriveTimer = setTimeout(() => loadRefmods(), 3000);
  const s = state.videoSettings;
  if (Array.isArray(s.refmods)) s.refmods = s.refmods.filter((m) => RM.files.includes(m.file));
  if (state.settingsOpen && state.tab === "video") render();
}
function refmodsField(s) {
  if (!RM.installed) return RM.loaded ? `<div class="field"><span class="label">RefMods</span><span class="hint">Saved references, made from pictures or video clips. They need the ComfyUI-MiniMaxH3Mod node pack: on RunPod, set ENABLE_REFMOD=true.</span></div>` : "";
  const chosen = s.refmods || [], left = RM.files.filter((f) => !chosen.some((m) => m.file === f));
  const making = RM.jobs.filter((j) => j.state === "making");
  return `<div class="field">
      <div class="field-head"><span class="label">RefMods</span><button class="link" data-act="makeRefmod">+ Make one</button></div>
      ${chosen.map((m) => `<div class="lora">
          <div class="lora-body">
            <span class="lora-name" title="${esc(m.file)}">${esc(m.file)}</span>
            <input type="range" min="0" max="1" step="0.05" value="${m.strength}" data-input="refmodStrength" data-id="${esc(m.file)}" aria-label="Strength of ${esc(m.file)}">
          </div>
          <span class="lora-val" data-refmod="${esc(m.file)}">${Number(m.strength).toFixed(2)}</span>
          <button class="lora-x" data-act="removeRefmod" data-id="${esc(m.file)}" aria-label="Remove ${esc(m.file)}">${ICON.x}</button>
        </div>`).join("")}
      ${chosen.length < 8 && left.length ? `<select class="select" data-change="addRefmod" aria-label="Add a RefMod"><option value="">+ Add a RefMod…</option>${left.map((f) => `<option value="${esc(f)}">${esc(f)}</option>`).join("")}</select>` : ""}
      ${making.map((j) => `<span class="hint">Making ${esc(j.name)}…</span>`).join("")}
      <span class="hint">${RM.files.length ? "Describe who is in the shot in the prompt: a RefMod has no tag." : "None yet: make one from pictures in the prompt bar, or from a video (its menu › Make a RefMod)."}</span>
      ${refmodDriveLine()}
    </div>`;
}
function refmodDriveLine() {
  const d = RM.drive || {};
  if (d.state === "off") return "";
  const text = d.state === "syncing" ? "Copying RefMods to and from Google Drive…"
    : d.state === "idle" ? "RefMods are kept in Google Drive › refmods too"
    : d.state === "error" ? "Couldn't copy RefMods to or from Google Drive: " + (d.error || "unknown error")
    : `Google Drive › refmods: ${d.drive ?? "?"} file${d.drive === 1 ? "" : "s"} · on this pod: ${d.here ?? "?"}` + (d.fetched ? ` · ${d.fetched} fetched` : "");
  return `<span class="hint ${d.state === "error" ? "bad-hint" : ""}" title="${esc(d.folder ? "This pod's RefMods folder: " + d.folder : "")}">${esc(text)}${d.state === "syncing" ? "" : ` · <button class="link" data-act="refmodSync">Sync now</button>`}</span>`;
}
const RM_KIND_HINTS = {
  identity: "Pictures of the person work best (a few angles). Full keeps the face better.",
  clothing: "Pictures of the outfit, or clips of someone wearing it.",
  style: "Clips or pictures in the style or look you want: colour, light, film grain, how it's shot. Compressed is enough.",
  background: "Pictures or clips of the place.",
  pose_motion: "Clips of the motion: a dance, a gesture, a camera move. For a move like waving, give 2 to 4 clips of different people doing it, and merge them. It guides the motion; it won't replay it exactly.",
  generic: "Pictures or clips of anything that should carry over: an effect, a mood, an object.",
};
function rmKindChanged() {
  const kind = $("rmKind").value;
  $("rmKindHint").textContent = RM_KIND_HINTS[kind] || "";
  rmCheckReady();
}
function rmCheckReady() {
  const pics = document.querySelectorAll("#rmPics input:checked").length, vids = document.querySelectorAll("#rmVids input:checked").length;
  $("rmSecsRow").hidden = !vids;
  // Merge and Only the movement are Compressed only: Full keeps each encode as it is.
  const compressed = document.querySelector("input[name=rmMode]:checked")?.value === "compressed";
  $("rmMergeRow").hidden = !(compressed && pics + vids > 1);
  $("rmMotionOnlyRow").hidden = !(compressed && vids && $("rmKind").value === "pose_motion");
  $("rmMergeBox").hidden = $("rmMergeRow").hidden && $("rmMotionOnlyRow").hidden;
  $("rmGo").disabled = !pics && !vids;
}
function openRefmodMaker(videoId = null) {
  const pics = state.attached.map((id) => state.refs.find((r) => r.id === id)).filter(Boolean);
  const vids = state.clips.filter((c) => kindOf(c) === "video" && c.status === "done" && c.file);
  const box = $("refmodModal");
  const first = videoId ? clipById(videoId) : null;
  $("rmName").value = first ? (first.prompt || String(first.imported || "clip").replace(/\.[^.]+$/, "")).replace(/[^A-Za-z0-9 _-]/g, "").trim().split(/\s+/).slice(0, 3).join("_").slice(0, 40)
    : pics[0] ? pics[0].name.replace(/[^A-Za-z0-9_-]/g, "").slice(0, 40) : "";
  $("rmDesc").value = "";
  $("rmKind").value = first ? "style" : "identity";
  $("rmPics").innerHTML = pics.length ? pics.map((r, i) => `<label class="rm-pic"><input type="checkbox" value="${r.id}" ${first ? "" : "checked"}>
      <img src="${r.img}" alt=""><span>Picture ${i + 1}</span></label>`).join("")
    : `<p class="hint">None in the prompt bar (Add image puts them there).</p>`;
  const max = RM.videosMax || 8;
  $("rmVids").innerHTML = vids.length ? vids.map((c) => `<label class="rm-pic" title="${esc(c.prompt || c.imported || "")}"><input type="checkbox" value="${c.id}" ${c.id === videoId ? "checked" : ""}>
      <video src="${clipUrl(c)}#t=0.5" muted playsinline preload="metadata"></video><span>${esc(clipLabel(c))}</span></label>`).join("")
    : `<p class="hint">No finished videos in the gallery yet. Import one from your computer with Import.</p>`;
  $("rmVids").onchange = () => {
    const on = document.querySelectorAll("#rmVids input:checked");
    if (on.length > max) { on[on.length - 1].checked = false; say(`At most ${max} clips.`); }
    rmCheckReady();
  };
  $("rmPics").onchange = rmCheckReady;
  $("rmKind").onchange = rmKindChanged;
  document.querySelectorAll("input[name=rmMode]").forEach((x) => { x.onchange = rmCheckReady; });
  $("rmMerge").checked = false; $("rmMotionOnly").checked = false;
  rmKindChanged();
  box.hidden = false;
  setTimeout(() => (first ? $("rmVids").querySelector("input:checked")?.scrollIntoView({ block: "nearest" }) : $("rmName").focus()), 30);
}
const clipLabel = (c) => (c.len ? Number(c.len).toFixed(1) + " s" : "Video") + (c.imported ? " · imported" : "");
async function makeRefmod(replace = false) {
  const pictures = [...document.querySelectorAll("#rmPics input:checked")].map((x) => x.value);
  const videos = [...document.querySelectorAll("#rmVids input:checked")].map((x) => x.value);
  const body = { name: $("rmName").value.trim(), pictures, videos, seconds: Number($("rmSecs").value) || 5,
    mode: document.querySelector("input[name=rmMode]:checked").value, concept: $("rmKind").value, desc: $("rmDesc").value.trim(), replace,
    merge: !$("rmMergeRow").hidden && $("rmMerge").checked, motionOnly: !$("rmMotionOnlyRow").hidden && $("rmMotionOnly").checked };
  if (!pictures.length && !videos.length) { say("Pick at least one picture or video clip."); return; }
  if (body.concept === "pose_motion" && !videos.length) { say("A motion needs at least one video clip."); return; }
  try {
    const r = await api("POST", "refmods", body);
    RM.jobs = r.jobs;
    $("refmodModal").hidden = true;
    say(`Making the RefMod ${r.job.name}. It's in ComfyUI's queue; it'll be in the list when it's done.`);
    render();
  } catch (error) {
    if (/already/.test(error.message) && !replace && (await askConfirm("Replace " + body.name + "?", error.message + " Make it again from these?", "Replace"))) return makeRefmod(true);
    say(error.message);
  }
}

// Refine faces (ComfyUI-H3-FaceRefine): a finished video's faces re-rendered up close and put back, as a new clip.
const FACES = { why: null, models: [], clip: null, face: null };
async function loadFaces() {
  try { Object.assign(FACES, await api("GET", "faces")); } catch {}
}
async function openFaces(c) {
  await loadFaces();
  if (FACES.why) { say(FACES.why); return; }
  if (!FACES.models.length) { say("Refine faces needs a MiniMax H3 model other than HyperFlow."); return; }
  FACES.clip = c.id;
  // The clip's own pictures first: one of them is usually the close-up of the face.
  const own = (c.pictures || c.refs || []).map((p) => p.id);
  FACES.face = null;
  const model = FACES.models.find((m) => m.name === c.model) || FACES.models[0];
  $("fcModel").innerHTML = FACES.models.map((m) => `<option value="${esc(m.name)}" ${m === model ? "selected" : ""}>${esc(m.name)} · ${m.steps} steps</option>`).join("");
  $("fcModelHint").textContent = c.model === model.name ? "The model the clip was made with."
    : c.profile === "hyperflow" ? `${c.model} can't refine (its fixed schedule can't re-render part of the way), so another model does the faces.`
    : `${c.model} isn't installed, so another model does the faces.`;
  FACES.order = [...own, ...state.refs.map((r) => r.id).filter((id) => !own.includes(id))];
  drawFacePics();
  document.querySelector("input[name=fcStrength][value=normal]").checked = true;
  $("facesModal").hidden = false;
}
function drawFacePics() {
  const pics = FACES.order.map((id) => state.refs.find((r) => r.id === id)).filter(Boolean);
  $("fcPics").innerHTML = pics.length ? pics.map((r) => `<button class="rm-pic fc-pic ${FACES.face === r.id ? "on" : ""}" data-act="facesPick" data-id="${r.id}" aria-pressed="${FACES.face === r.id}">
      <img src="${r.img}" alt=""><span>${esc(r.name)}</span></button>`).join("")
    : `<p class="hint">No pictures in Add image. Without one, the largest face is refined.</p>`;
  $("fcFaceHint").textContent = FACES.face ? "This face is refined, looking like the picture: a sharp close-up works best."
    : "None picked: the largest face is refined, from the clip's own pictures and prompt.";
}
async function refineFaces() {
  const body = { model: $("fcModel").value, strength: document.querySelector("input[name=fcStrength]:checked").value,
    face: FACES.face };
  try {
    const r = await api("POST", "clips/" + FACES.clip + "/faces", body);
    $("facesModal").hidden = true;
    upsertClip(r.clip);
    say("Refining the faces. The result comes in as a new clip next to the original.");
  } catch (error) { say(error.message); }
}

// Swap (Alissonerdx's Character Swap v1.1 and Head Swap v1.0): a video as the guide, a picture as the new character.
const SWAP = { clip: null, mode: "character", picture: null, res: 0.6, status: null,
  who: "main", point: null, opacity: 1, check: null };   // check: {state, pid, url, error} of the marked preview
async function openSwap(c) {
  try { SWAP.status = await api("GET", "swap"); } catch (error) { say(error.message); return; }
  if (SWAP.clip !== c.id) Object.assign(SWAP, { who: "main", point: null, check: null });
  SWAP.clip = c.id;
  if (!state.refs.some((r) => r.id === SWAP.picture)) SWAP.picture = null;
  $("swFrame").src = clipUrl(c);
  $("swDetails").value = "";
  const models = SWAP.status.models;
  if (!models.some((m) => m.id === SWAP.model)) SWAP.model = models[0]?.id || null;
  $("swModel").innerHTML = models.map((m) => `<option value="${m.id}" ${m.id === SWAP.model ? "selected" : ""}>${esc(m.name)}${m.id === SWAP.status.base ? " · the author's" : " · untested"}</option>`).join("");
  drawSwap();
  $("swapModal").hidden = false;
}
function drawSwap() {
  const st = SWAP.status, head = SWAP.mode === "head";
  const why = st.modes[SWAP.mode] || (st.models.length ? null : "Swap needs a MiniMax H3 model other than HyperFlow: MiniMax H3 + H3 Turbo is the author's.");
  const m = st.models.find((x) => x.id === SWAP.model);
  $("swModelHint").textContent = !m ? "" : m.id === st.base ? "The author's: MiniMax H3's own model, which the swap LoRAs were trained on, with the DMD 8-step turbo LoRA as in his workflow."
    : `Untested: the swap LoRAs were trained on MiniMax H3's own model. ${m.turbo ? `${m.name} runs its own way (${m.steps} steps), without the DMD turbo LoRA.` : `${m.name} gets the DMD turbo LoRA and 8 steps, like the author's.`}`;
  $("swModes").innerHTML = [["character", "Character", "the whole person: face, hair, body and clothes"], ["head", "Head only", "experimental: the head, on the video's own body"]]
    .map(([id, name, what]) => `<button class="${SWAP.mode === id ? "on" : ""}" data-act="swapMode" data-id="${id}" title="${what}">${name}</button>`).join("");
  $("swModeHint").textContent = why || (head ? "Replaces only the head; the body, clothes and motion stay the video's. Experimental."
    : "The person in the picture takes the place of the video's main person, with the same motion, lips and camera.");
  const pics = state.refs;
  $("swPics").innerHTML = pics.length ? pics.map((r) => `<button class="rm-pic fc-pic ${SWAP.picture === r.id ? "on" : ""}" data-act="swapPick" data-id="${r.id}" aria-pressed="${SWAP.picture === r.id}">
      <img src="${r.img}" alt=""><span>${esc(r.name)}</span></button>`).join("")
    : `<p class="hint">Add the picture with Add image first.</p>`;
  $("swPicHint").textContent = (head ? "A clear picture of the face and hair." : "The whole person, head to toe if you can.") +
    (st.background ? " Its background is replaced by grey, as in the author's workflow." : "");
  $("swDetailsLabel").textContent = head ? "The face in words" : "The character in words";
  $("swDetails").placeholder = (head ? "Male, oval face shape, light brown skin, dark eyes, black tousled hair" : "Female, tall, slim, long red hair, green eyes, black leather jacket, blue jeans, white sneakers");
  $("swDetailsHint").textContent = st.describe ? "Optional. Empty: written from the picture while it renders, and saved with the clip."
    : "Needed: describing the picture needs MOSS-Image's text encoder, which isn't installed.";
  const pickable = !head && st.mark;
  $("swWhoField").hidden = head;
  if (!pickable && SWAP.who === "pick") SWAP.who = "main";
  $("swWho").innerHTML = [["main", "The main person"], ["pick", "Pick on the first frame"]]
    .map(([id, name]) => `<button class="${SWAP.who === id ? "on" : ""}" data-act="swapWho" data-id="${id}" ${id === "pick" && !pickable ? "disabled" : ""}>${name}</button>`).join("");
  $("swPick").hidden = SWAP.who !== "pick";
  $("swWhoHint").textContent = SWAP.who === "main" ? "Character Swap replaces the person it sees as the main one, usually the biggest and most central."
    + (st.mark ? "" : " Picking a person needs SAM 3.1 (gear menu › Models › Swap).")
    : "Click the person to replace. They're painted red through the clip and the prompt says to replace only them; everyone else stays.";
  const dot = $("swDot");
  dot.hidden = !SWAP.point;
  if (SWAP.point) { const r = swFrameRect(); dot.style.left = r.left + SWAP.point.x * r.width + "px"; dot.style.top = r.top + SWAP.point.y * r.height + "px"; }
  $("swOpacity").innerHTML = [[1, "Solid"], [0.5, "See-through"]].map(([v, name]) =>
    `<button class="${SWAP.opacity === v ? "on" : ""}" data-act="swapOpacity" data-id="${v}">${name}</button>`).join("");
  const ck = SWAP.check;
  $("swCheck").disabled = !SWAP.point || ck?.state === "working";
  $("swCheckOut").innerHTML = !ck ? "" : ck.state === "working" ? `<span class="hint">Marking the person through the clip…</span>`
    : ck.state === "failed" ? `<span class="hint warn">${esc(ck.error || "It didn't work.")}</span>`
    : `<video src="${ck.url}" autoplay muted loop playsinline></video><span class="hint">The red person is the one that's replaced. Wrong one? Click again.</span>`;
  $("swRes").innerHTML = st.megapixels.map((mp) => `<button class="${SWAP.res === mp ? "on" : ""}" data-act="swapRes" data-id="${mp}">${mp.toFixed(1)} MP</button>`).join("");
  $("swGo").disabled = !!why || !SWAP.picture || (SWAP.who === "pick" && !SWAP.point);
}
function swapMark() {
  return !$("swWhoField").hidden && SWAP.who === "pick" && SWAP.point ? { ...SWAP.point, opacity: SWAP.opacity } : null;
}
async function checkMark() {
  SWAP.check = { state: "working" };
  drawSwap();
  try {
    const { prompt_id } = await api("POST", "clips/" + SWAP.clip + "/swap-mark", { mark: swapMark(), res: SWAP.res });
    SWAP.check.pid = prompt_id;
    while (SWAP.check?.pid === prompt_id) {
      await new Promise((r) => setTimeout(r, 2000));
      const r = await api("GET", "swap-mark/" + prompt_id);
      if (SWAP.check?.pid !== prompt_id || r.state === "working") continue;
      SWAP.check = r.state === "done" ? { state: "done", url: fileUrl(r.subfolder, r.file) } : { state: "failed", error: r.error };
      drawSwap();
    }
  } catch (error) { SWAP.check = { state: "failed", error: error.message }; drawSwap(); }
}
// The picture inside the frame box (object-fit: contain leaves bars), in pixels from the box's corner.
function swFrameRect() {
  const v = $("swFrame"), w = v.clientWidth, h = v.clientHeight;
  const scale = Math.min(w / (v.videoWidth || w), h / (v.videoHeight || h));
  const pw = (v.videoWidth || w) * scale, ph = (v.videoHeight || h) * scale;
  return { left: (w - pw) / 2, top: (h - ph) / 2, width: pw, height: ph };
}
$("swFrameBox").addEventListener("click", (e) => {
  const box = $("swFrame").getBoundingClientRect(), r = swFrameRect();
  const x = (e.clientX - box.left - r.left) / r.width, y = (e.clientY - box.top - r.top) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  SWAP.point = { x: Math.round(x * 10000) / 10000, y: Math.round(y * 10000) / 10000 };
  SWAP.check = null;
  drawSwap();
});
async function runSwap() {
  const body = { model: SWAP.model, mode: SWAP.mode, picture: SWAP.picture, res: SWAP.res, details: $("swDetails").value.trim(), mark: swapMark() };
  try {
    const r = await api("POST", "clips/" + SWAP.clip + "/swap", body);
    $("swapModal").hidden = true;
    upsertClip(r.clip);
    say("Swapping. The result comes in as a new clip next to the video.");
  } catch (error) { say(error.message); }
}

// The Klein tab's settings: Model, Size (Match image 1 or an aspect ratio), Resolution, Steps, Guidance (Base
// only: Distilled renders without it), Seed, LoRAs (the loras/Flux Klein folder).
// Face swap's own settings: the BFS LoRA, and with Only the head its sampler, the mask's edge and Refine face.
function swapSettingsHTML(s) {
  const loras = OPT.klein.swapLoras || [], ho = headOnly(), lan = ho && s.swapSampler === "lanpaint";
  const slider = (key, label, min, max, value, text) => `<div class="field">
      <div class="field-head"><label class="label" for="sw-${key}">${label}</label><span class="value" data-val="${key}">${text}</span></div>
      <input id="sw-${key}" type="range" min="${min}" max="${max}" step="1" value="${value}" data-input="swapNum" data-id="${key}"></div>`;
  return `
    <div class="field">
      <label class="label" for="swapLora">Face swap LoRA</label>
      ${loras.length > 1 ? `<select class="select" id="swapLora" data-change="swapLora">${loras.map((l) =>
        `<option value="${esc(l.file)}" ${l.file === (s.swapLora || loras[0].file) ? "selected" : ""}>${esc(l.name)}</option>`).join("")}</select>`
        : `<div class="fixed-field">${loras.length ? esc(loras[0].name) : "Not installed"}</div>`}
      <span class="hint">Alissonerdx's BFS head swap for Klein 9B, at 1.0. Rank 128 keeps more detail; rank 64 is the one Comfy's workflow uses.</span>
    </div>
    ${ho ? `<div class="field">
      <span class="label">Sampler (Only the head)</span>
      <div class="seg">
        <button class="${!lan ? "on" : ""}" data-act="swapSampler" data-id="standard">Standard</button>
        <button class="${lan ? "on" : ""}" data-act="swapSampler" data-id="lanpaint" ${OPT.klein.lanpaint ? "" : "disabled"}>LanPaint</button>
      </div>
      <span class="hint">${OPT.klein.lanpaint ? "LanPaint blends the new head into the untouched picture better, and is slower (it thinks twice a step)." : "LanPaint isn't installed (scraed/LanPaint in custom_nodes)."}</span>
    </div>
    ${lan ? slider("lanSteps", "LanPaint steps", 8, 40, s.lanSteps, s.lanSteps) : ""}
    ${slider("maskGrow", "Head mask: grow", 0, 64, s.maskGrow, s.maskGrow + " px")}
    ${slider("maskFeather", "Head mask: soft edge", 0, 64, s.maskFeather, s.maskFeather + " px")}
    ${s.refine ? slider("refineStrength", "Refine face: strength", 20, 90, Math.round(s.refineStrength * 100), Math.round(s.refineStrength * 100) + "%") : ""}`
    : `<span class="hint">Turn on <b>Only the head</b> in the prompt bar to keep everything but the head exactly as it is, and for LanPaint and Refine face.</span>`}`;
}

function kleinSettingsHTML(s, model, T, minSteps, stepSize, recPct) {
  const r1 = slotRef(0), [w, h] = kleinSize(s), swap = swapOn(), res = swap ? s.swapRes : s.res;
  const sizeOf = (mp) => kleinSize({ ...s, res: mp, swapRes: mp }).join("×");
  return `
    <div class="field">
      <label class="label" for="model">Model</label>
      <select class="select" id="model" data-change="model">
        ${(T.models.length ? T.models : [model]).map((m) => `<option value="${esc(m.name)}" ${m.name === s.model ? "selected" : ""}>${esc(m.name)}</option>`).join("")}
      </select>
      <span class="hint">${model.base ? "Undistilled: slower and more varied, with guidance." : "Distilled: fast, 4 steps, no guidance."}</span>
    </div>

    ${swap ? swapSettingsHTML(s) : ""}

    <div class="field" ${swap ? "hidden" : ""}>
      <span class="label">Size</span>
      <div class="aspects five klein-sizes">
        <button class="match ${s.size === "match" ? "on" : ""}" data-act="kleinMatch">${swapOn() ? "Match the body" : "Match image 1"}<small>${r1 ? (r1.w ? r1.w + "×" + r1.h : slotWord(0)) : slotWord(0) + " is empty: 1:1"}</small></button>
        ${Object.keys(T.aspects).map((a) => {
          const [aw, ah] = T.aspects[a];
          const k = 24 / Math.max(aw, ah);
          return `<button class="${s.size !== "match" && a === s.aspect ? "on" : ""}" data-act="aspect" data-id="${a}"><i style="width:${Math.round(aw * k)}px;height:${Math.round(ah * k)}px"></i>${a}</button>`;
        }).join("")}
      </div>
    </div>

    <div class="field">
      <label class="label" for="res">Resolution</label>
      <select class="select" id="res" data-change="res">
        ${T.megapixels.map((mp) => `<option value="${mp}" ${mp === res ? "selected" : ""}>${mpText(mp)} (${sizeOf(mp)})${(swap ? [2.0] : T.recommendedMP || []).includes(mp) ? " · recommended" : ""}</option>`).join("")}
      </select>
      <span class="hint">Renders at ${w}×${h}${swap ? ", the body's shape. The pictures go in at this size too (the workflow uses 2 MP)" : s.size === "match" && r1 ? ", image 1's shape" : ""}.</span>
    </div>

    <div class="field" ${swap && headOnly() && s.swapSampler === "lanpaint" ? "hidden" : ""}>
      <div class="field-head"><label class="label" for="steps">Steps</label><button class="value value-edit ${s.steps === model.steps ? "rec" : ""}" id="stepsVal" data-act="editValue" data-id="steps" title="Click to type a number of steps">${stepsText(s.steps, model.steps)}</button></div>
      <div class="steps-wrap">
        <input id="steps" type="range" min="${minSteps}" max="${model.maxSteps}" step="${stepSize}" value="${s.steps}" data-input="steps" aria-describedby="stepsRec">
        <span class="rec-mark" style="left:calc(${recPct}% + ${(0.5 - recPct / 100) * 16}px)" aria-hidden="true"></span>
      </div>
      <div class="field-head steps-foot"><span class="hint" id="stepsRec">Recommended for ${esc(model.name)}: ${model.steps}</span>
        <button class="link" data-act="recSteps" id="recBtn" style="visibility:${s.steps === model.steps ? "hidden" : "visible"}">Use ${model.steps}</button></div>
    </div>

    ${model.base ? guidanceField(s, model) : ""}

    <div class="field">
      <span class="label">Seed</span>
      <div class="seed-row">
        <div class="seg">
          <button class="${s.seedMode === "random" ? "on" : ""}" data-act="seedMode" data-id="random">Random</button>
          <button class="${s.seedMode === "fixed" ? "on" : ""}" data-act="seedMode" data-id="fixed">Fixed</button>
        </div>
        ${s.seedMode === "fixed"
          ? `<input class="seed-input" inputmode="numeric" aria-label="Seed number" value="${s.seed}" data-change="seed">
             <button class="icon-btn" data-act="rollSeed" aria-label="New random seed">${ICON.dice}</button>`
          : `<span class="seed-note">A new seed every run, saved with each image.</span>`}
      </div>
    </div>

    ${lorasField(s)}`;
}

// The Pruna tab's settings: Model, Aspect ratio (Match picture 1) or the sheet's shape and background,
// Resolution (1-8 MP), Steps (always 8), Seed, LoRAs (the loras/P-H3 Image folder).
// The Pruna tab runs P-H3-Image and MiniMax's own H3-Image, each with its own sizes.
const h3img = (m) => !!(m && m.h3image);
const prunaMPs = (m) => (m && m.megapixels) || OPT.pruna.megapixels;
// A size the model has: the nearest one (P-H3 goes to 8 MP, H3-Image from 4 to 16).
const prunaFit = (m, v) => { const l = prunaMPs(m); return l.includes(v) ? v : l.reduce((a, b) => (Math.abs(b - v) < Math.abs(a - v) ? b : a)); };

function prunaSettingsHTML(s, model, T) {
  const r1 = slotRef(0, state.prunaSlots), [w, h] = prunaSize(s), mp = sheetOn() ? s.sheetRes : s.res;
  const mps = prunaMPs(model), recMP = sheetOn() ? (h3img(model) ? 16 : 8) : model.recommendedMP || 4;
  const sizeOf = (v) => prunaSize({ ...s, res: v, sheetRes: v }).join("×");
  return `
    <div class="field">
      <label class="label" for="model">Model</label>
      ${T.models.length > 1 ? `<select class="select" id="model" data-change="model">
        ${T.models.map((m) => `<option value="${esc(m.name)}" ${m.name === s.model ? "selected" : ""}>${esc(m.name)}</option>`).join("")}
      </select>` : `<div class="fixed-field">${esc(model.name)}</div>`}
      <span class="hint">${h3img(model) ? "MiniMax's own image model: 6 steps, made for 4 to 16 MP. It reads your prompt as written."
        : model.id === "lora" ? "MiniMax H3's FL2VA model with Pruna's P-H3-Image LoRA." : "Pruna's full P-H3-Image model."}</span>
    </div>

    ${sheetOn() ? `<div class="field">
      <span class="label">Sheet shape</span>
      <div class="fixed-field">${sheetLayout().ratioText}<small>set by the views you pick</small></div>
    </div>
    <div class="field">
      <span class="label">Background</span>
      <div class="seg">${SHEET_BGS.map((b) => `<button class="${state.prunaSheet.bg === b ? "on" : ""}" data-act="sheetBg" data-id="${b}">${b[0].toUpperCase() + b.slice(1)}</button>`).join("")}</div>
    </div>` : `<div class="field">
      <span class="label">Aspect ratio</span>
      <div class="aspects five klein-sizes">
        <button class="match ${s.size === "match" ? "on" : ""}" data-act="sizeMatch">Match picture 1<small>${r1 ? (r1.w ? r1.w + "×" + r1.h : "picture 1") : "picture 1 is empty: 1:1"}</small></button>
        ${Object.keys(T.aspects).map((a) => {
          const [aw, ah] = T.aspects[a];
          const k = 24 / Math.max(aw, ah);
          return `<button class="${s.size !== "match" && a === s.aspect ? "on" : ""}" data-act="aspect" data-id="${a}"><i style="width:${Math.round(aw * k)}px;height:${Math.round(ah * k)}px"></i>${a}</button>`;
        }).join("")}
      </div>
    </div>`}

    <div class="field">
      <label class="label" for="res">Resolution</label>
      <select class="select" id="res" data-change="res">
        ${mps.map((v) => `<option value="${v}" ${v === mp ? "selected" : ""}>${mpText(v)} (${sizeOf(v)})${v === recMP ? " · recommended" : ""}</option>`).join("")}
      </select>
      <span class="hint">Renders at ${w}×${h}. ${h3img(model) ? "16 MP is MiniMax's sweet spot; below 4 MP it looks soft. On a 32 GB card, 4 to 8 MP."
        : sheetOn() ? "8 MP keeps the faces sharp on a wide sheet." : "8 MP is the model's own size, and the slowest."}</span>
    </div>

    <div class="field">
      <span class="label">Steps</span>
      <div class="fixed-field">${model.steps || OPT.pruna.steps || 8}<small>${h3img(model) ? "the model is made for 6 steps, at guidance 1" : "a fixed schedule: more steps don't help"}</small></div>
    </div>

    <div class="field">
      <span class="label">Seed</span>
      <div class="seed-row">
        <div class="seg">
          <button class="${s.seedMode === "random" ? "on" : ""}" data-act="seedMode" data-id="random">Random</button>
          <button class="${s.seedMode === "fixed" ? "on" : ""}" data-act="seedMode" data-id="fixed">Fixed</button>
        </div>
        ${s.seedMode === "fixed"
          ? `<input class="seed-input" inputmode="numeric" aria-label="Seed number" value="${s.seed}" data-change="seed">
             <button class="icon-btn" data-act="rollSeed" aria-label="New random seed">${ICON.dice}</button>`
          : `<span class="seed-note">A new seed every run, saved with each image.</span>`}
      </div>
    </div>

    ${model.loras === false ? `<div class="field"><span class="label">LoRAs</span><span class="hint">MiniMax H3's LoRAs don't fit H3-Image, so it renders without them.</span></div>` : lorasField(s)}`;
}

// Sampler and scheduler. "Model default" (the empty choice) uses what the model is tuned for.
function samplingFields(s, model) {
  const hyperflow = model.scheduler === "hyperflow";
  const schedulers = (OPT.schedulers || []).concat(isImage() || !OPT.beta57 ? [] : [OPT.beta57]);
  const opt = (value, current, label) => `<option value="${esc(value)}" ${value === current ? "selected" : ""}>${esc(label || value)}</option>`;
  return `<div class="field sampling">
      <div class="sampling-row">
        <div><label class="label" for="sampler">Sampler</label>
          <select class="select" id="sampler" data-change="sampler">
            ${opt("", s.sampler || "", "Model default (" + (model.sampler || "?") + ")")}
            ${(OPT.samplers || []).map((n) => opt(n, s.sampler || "")).join("")}
          </select></div>
        <div><label class="label" for="scheduler">Scheduler</label>
          <select class="select" id="scheduler" data-change="scheduler" ${hyperflow ? "disabled" : ""}>
            ${hyperflow ? opt("", "", "HyperFlow's own")
              : opt("", s.scheduler || "", "Model default (" + (model.scheduler || "?") + ")") + schedulers.map((n) => opt(n, s.scheduler || "")).join("")}
          </select></div>
      </div>
      ${hyperflow ? `<span class="hint">HyperFlow samples on its own 4, 6 or 8-step schedule.</span>` : ""}
    </div>`;
}

// Realism: the MMH3 Realism Slider LoRA on every video, joining after the first steps so the composition stays.
const realismOf = (s) => (s.realism ||= { on: true, strength: OPT.realism?.strength || 1.2 });
function realismField(s) {
  const R = OPT.realism || {}, r = realismOf(s);
  if (!R.file) return `<div class="field"><span class="label">Realism</span><span class="hint">Natural skin, with the MMH3 Realism Slider LoRA: put MMH3-Realism-V2.safetensors in models/loras (on RunPod, REALISM_LORA=true).</span></div>`;
  return `<div class="field">
      <span class="label">Realism</span>
      <div class="lora ${r.on ? "" : "dim"}">
        <button class="switch ${r.on ? "on" : ""}" data-act="toggleRealism" aria-label="${r.on ? "Switch off" : "Switch on"} Realism" aria-pressed="${r.on}"></button>
        <div class="lora-body">
          <span class="lora-name" title="${esc(R.file)}">${esc(loraStem(R.file))}</span>
          <input type="range" min="${R.range[0]}" max="2" step="0.05" value="${r.strength}" data-input="realismStrength" aria-label="Strength of Realism">
        </div>
        <span class="lora-val" id="realismVal">${Number(r.strength).toFixed(2)}</span>
      </div>
      <span class="hint">Natural, un-airbrushed skin. It joins after the first steps, so the composition stays as it would be. ${R.strength} is the H3 editor's setting.</span>
    </div>`;
}
function lorasField(s) {
  return `<div class="field">
      <div class="field-head"><span class="label">LoRAs</span><button class="link" data-act="loraPicker">+ Add LoRA</button></div>
      ${s.loras.map((l) => `<div class="lora ${l.on ? "" : "dim"}">
          <button class="switch ${l.on ? "on" : ""}" data-act="toggleLora" data-id="${esc(l.id)}" aria-label="${l.on ? "Switch off" : "Switch on"} ${esc(l.name)}" aria-pressed="${l.on}"></button>
          <div class="lora-body">
            <span class="lora-name" title="${esc(l.name)}">${esc(loraStem(l.name))}</span>
            <input type="range" min="0" max="2" step="0.05" value="${l.strength}" data-input="loraStrength" data-id="${esc(l.id)}" aria-label="Strength of ${esc(l.name)}">
          </div>
          <span class="lora-val" data-strength="${esc(l.id)}">${Number(l.strength).toFixed(2)}</span>
          <button class="lora-x" data-act="removeLora" data-id="${esc(l.id)}" aria-label="Remove ${esc(l.name)}">${ICON.x}</button>
        </div>`).join("")}
    </div>`;
}

// Guidance (CFG): how closely MOSS-Image follows the prompt. The recommended value is marked on the slider.
function guidanceField(s, model) {
  const g = (isKlein() ? OPT.klein.cfg : OPT.image.cfg) || OPT.image.cfg, rec = model.cfg;
  const pct = ((rec - g.min) / (g.max - g.min)) * 100;
  return `<div class="field">
      <div class="field-head"><label class="label" for="cfg">Guidance</label><button class="value value-edit ${s.cfg === rec ? "rec" : ""}" id="cfgVal" data-act="editValue" data-id="cfg" title="Click to type a guidance value">${cfgText(s.cfg, rec)}</button></div>
      <div class="steps-wrap">
        <input id="cfg" type="range" min="${g.min}" max="${g.max}" step="${g.step}" value="${s.cfg}" data-input="cfg" aria-describedby="cfgRec">
        <span class="rec-mark" style="left:calc(${pct}% + ${(0.5 - pct / 100) * 16}px)" aria-hidden="true"></span>
      </div>
      <div class="field-head steps-foot"><span class="hint" id="cfgRec">How closely the image follows the prompt. Recommended: ${rec}</span>
        <button class="link" data-act="recCfg" id="cfgBtn" style="visibility:${s.cfg === rec ? "hidden" : "visible"}">Use ${rec}</button></div>
    </div>`;
}
const cfgText = (n, rec) => Number(n).toFixed(1) + (n === rec ? " · recommended" : "");

/* Clip page */
// Full screen covers the clip page and hides everything but the player, so redrawing keeps it.
const isFull = () => document.fullscreenElement === $("clipView");
let shownClip = null;   // which clip (and in which state) the player shows, so it is only rebuilt when that changes

function renderClip() {
  const c = clipById(state.openId);
  if (!c) { state.view = "gallery"; render(); return; }
  $("clipView").classList.toggle("fs", isFull());
  const key = c.id + ":" + c.status + ":" + (c.file || c.url || "") + ":" + Boolean(isBusy(c) && c.preview) + ":" + Boolean(c.swap && state.compare);
  if (shownClip !== key && kindOf(c) === "image") {
    shownClip = key;
    $("clipView").innerHTML = `
      <div class="clip-main">
        <div class="player still" data-clip="${c.id}">
          <span class="pic" style="${picStyle(shapeOf(c), toneOf(c.id))}">${isBusy(c) && c.preview ? `<img data-live="${c.id}" src="${c.preview}" alt="">` : c.status === "done" ? `<img id="pimg" src="${clipUrl(c)}" alt="${esc(c.prompt)}">${compareImg(c)}` : ""}</span>
          ${isBusy(c)
            ? `<span class="busy ${c.preview ? "has-live" : ""}"><span data-progtext="${c.id}">${progressText(c)}</span><span class="track"><span class="fill" data-prog="${c.id}" style="width:${barWidth(c)}"></span></span>
                <button class="cancel" data-act="cancel" data-id="${c.id}">Cancel render</button></span>`
            : c.status === "failed"
              ? `<span class="busy failed"><span>Render failed</span><span class="fail-why">${esc(c.error || "")}</span></span>`
              : `<button class="fs-btn still-fs" data-act="fullscreen" aria-label="Full screen">${ICON.expand}</button>`}
          ${compareHTML(c)}
        </div>
        <span class="label takes-label">Other images from this prompt</span>
        <div class="takes" id="takes"></div>
      </div>
      <aside class="clip-side" id="clipSide"></aside>`;
  } else if (shownClip !== key) {
    shownClip = key;
    $("clipView").innerHTML = `
      <div class="clip-main">
        <div class="player" data-clip="${c.id}">
          <span class="pic" style="${picStyle(c.aspect, toneOf(c.id))}">${isBusy(c) && c.preview ? liveVideo(c) : c.file ? `<video id="pv" src="${clipUrl(c)}" loop playsinline preload="auto"></video>` : ""}</span>
          ${isBusy(c)
            ? `<span class="busy ${c.preview ? "has-live" : ""}"><span data-progtext="${c.id}">${progressText(c)}</span><span class="track"><span class="fill" data-prog="${c.id}" style="width:${barWidth(c)}"></span></span>
                <button class="cancel" data-act="cancel" data-id="${c.id}">Cancel render</button></span>`
            : c.status === "failed"
              ? `<span class="busy failed"><span>Render failed</span><span class="fail-why">${esc(c.error || "")}</span></span>`
              : `<div class="controls">
                  <button class="play" id="playBtn" data-act="play" aria-label="Play">${ICON.play}</button>
                  <span class="scrub" id="scrub"><span id="playFill" style="width:0%"></span></span>
                  <span class="time" id="playTime">0:00 / ${clock(c.len)}</span>
                  <button class="fs-btn" data-act="fullscreen" aria-label="Full screen">${ICON.expand}</button>
                </div>`}
        </div>
        <span class="label takes-label">Takes of this prompt</span>
        <div class="takes" id="takes"></div>
      </div>
      <aside class="clip-side" id="clipSide"></aside>`;
    wirePlayer();
    $("pv")?.play().catch(() => {});
  }
  renderTakes(c);
  renderClipSide(c);
  syncFullscreenButton();
}

// A video imported as it is, without a prompt or settings: nothing to reuse, and no takes.
const plainVideo = (c) => c.found === "video file";

function renderTakes(c) {
  document.querySelector(".takes-label")?.toggleAttribute("hidden", plainVideo(c));
  const takes = plainVideo(c) ? [] : state.clips.filter((x) => x.prompt === c.prompt && tabOf(x) === tabOf(c));
  $("takes").innerHTML = takes.map((x) => `<button class="take ${x.id === c.id ? "on" : ""}" data-act="open" data-id="${x.id}">
      <i style="background:${toneOf(x.id)}">${kindOf(x) === "image" ? (x.status === "done" ? `<img src="${clipUrl(x)}" alt="">` : "") : x.file ? `<video src="${clipUrl(x)}" muted preload="metadata"></video>` : ""}</i>${mpText(x.res)} · Seed ${x.seed}<span data-takeprog="${x.id}">${takeProgress(x)}</span>
    </button>`).join("");
}

function renderClipSide(c) {
  if (kindOf(c) === "image") { renderImageSide(c); return; }
  if (plainVideo(c)) {
    $("clipSide").innerHTML = `
    <dl class="facts">
      <dt>Size</dt><dd>${c.width}×${c.height} · ${c.aspect}</dd>
      <dt>Length</dt><dd>${c.len} s</dd>
    </dl>
    <p class="hint imported">Imported from ${esc(c.imported)}, as it is: it has no prompt or settings. Swap a character into it, or put it in the edit.</p>
    <div class="actions">
      <button class="btn primary" data-act="swapThis" ${c.file ? "" : "disabled"}>${ICON.swap}Swap character…</button>
      <div class="actions-grid">
        <button class="btn" data-act="download" ${c.file ? "" : "disabled"}>Download</button>
        <button class="btn danger" data-act="deleteClip">Delete</button>
      </div>
    </div>`;
    return;
  }
  $("clipSide").innerHTML = `
    <div class="field">
      <span class="label">Prompt</span>
      <div class="prompt-box"><p class="clip-prompt">${promptHTML(c.prompt, "video", undefined, c.pictures ? c.pictures.length : null)}</p>
        <button class="copy prompt-copy" data-act="copyPrompt" aria-label="Copy the prompt" title="Copy the prompt">${ICON.copy}</button></div>
    </div>
    <dl class="facts">
      <dt>Model</dt><dd>${esc(c.model)}</dd>
      <dt>Steps</dt><dd>${c.steps}</dd>
      <dt>Resolution</dt><dd>${mpText(c.res)} (${c.width}×${c.height})</dd>
      <dt>Aspect</dt><dd>${c.aspect}</dd>
      <dt>Length</dt><dd>${c.len} s · ${snapFrames(c.len)} frames</dd>
      <dt>Seed</dt><dd class="seed-dd">${c.seed}<button class="copy" data-act="copySeed" aria-label="Copy seed">${ICON.copy}</button></dd>
      <dt>LoRAs</dt><dd>${c.loras.length ? c.loras.map((l) => esc(loraText(l))).join("<br>") : '<span class="label">None</span>'}</dd>
      ${samplingFacts(c)}
    </dl>
    ${c.imported ? `<p class="hint imported">Imported from ${esc(c.imported)}${c.found === "graph" ? ". Settings worked out from its ComfyUI workflow" : ""}.</p>` : ""}
    <div class="actions">
      <button class="btn primary" data-act="reuse">Use this prompt and settings</button>
      <button class="btn" data-act="keepSeed">Keep this seed</button>
      <div class="actions-grid">
        <button class="btn" data-act="download" ${c.file ? "" : "disabled"}>Download</button>
        <button class="btn" data-act="exportZip" ${c.file && c.status === "done" ? "" : "disabled"} title="The video and its reference images in one .zip. Import it here or in another MiniMax H3 Vision.">Export .zip</button>
        <button class="btn danger wide" data-act="deleteClip">Delete</button>
      </div>
    </div>`;
}

// A clip's sampler and scheduler: its own choice, or (in grey) its model's.
function samplingFacts(c) {
  const models = kindOf(c) === "image" ? OPT.image.models : OPT.models;
  // An edit's defaults are the edit LoRA's (euler/normal), whatever the checkpoint.
  const m = c.edit && OPT.image.edit ? OPT.image.edit : models.find((x) => x.name === c.model) || {};
  const one = (own, dflt) => own ? esc(own) : dflt ? `${esc(dflt === "hyperflow" ? "HyperFlow's own" : dflt)} <span class="label">· model default</span>` : '<span class="label">Model default</span>';
  return `<dt>Sampler</dt><dd>${one(c.sampler, m.sampler)}</dd><dt>Scheduler</dt><dd>${one(c.scheduler, m.scheduler)}</dd>`;
}
// Images saved before the negative prompt box existed used the model's own.
const negativeDefault = (modelName) => (OPT.image.models.find((m) => m.name === modelName) || OPT.image.models[0] || {}).negative || "";

function renderImageSide(c) {
  if (c.family === "klein") { renderKleinSide(c); return; }
  if (c.family === "ph3") { renderPrunaSide(c); return; }
  $("clipSide").innerHTML = `
    <div class="field">
      <span class="label">Prompt</span>
      <div class="prompt-box"><p class="clip-prompt">${esc(c.prompt)}</p>
        <button class="copy prompt-copy" data-act="copyPrompt" aria-label="Copy the prompt" title="Copy the prompt">${ICON.copy}</button></div>
    </div>
    ${c.edit ? editedFrom(c) : ""}
    <dl class="facts">
      <dt>Model</dt><dd>${esc(c.model)}${c.edit ? " · Edit" : ""}</dd>
      <dt>Steps</dt><dd>${c.steps}</dd>
      <dt>Guidance</dt><dd>${Number(c.cfg).toFixed(1)}</dd>
      <dt>Resolution</dt><dd>${mpText(c.res)} (${c.width}×${c.height})</dd>
      <dt>Aspect</dt><dd>${c.edit ? "The picture's shape" : c.aspect}</dd>
      <dt>Seed</dt><dd class="seed-dd">${c.seed}<button class="copy" data-act="copySeed" aria-label="Copy seed">${ICON.copy}</button></dd>
      <dt>LoRAs</dt><dd>${c.loras.length ? c.loras.map((l) => esc(loraText(l))).join("<br>") : '<span class="label">None</span>'}</dd>
      ${samplingFacts(c)}
      <dt>Negative</dt><dd>${c.negative != null ? (c.negative ? esc(c.negative) : '<span class="label">None</span>') : esc(negativeDefault(c.model)) || '<span class="label">None</span>'}</dd>
    </dl>
    ${c.imported ? `<p class="hint imported">Imported from ${esc(c.imported)}${c.found === "graph" ? ". Settings worked out from its ComfyUI workflow" : ""}.</p>` : ""}
    <div class="actions">
      <button class="btn accent" data-act="useInVideo" ${c.status === "done" ? "" : "disabled"}>${ICON.video}Use in a video</button>
      <button class="btn" data-act="editThis" ${c.status === "done" ? "" : "disabled"}>${ICON.klein}Edit in Klein</button>
      <button class="btn" data-act="editInPruna" ${c.status === "done" ? "" : "disabled"}>${ICON.pruna}Edit in Pruna</button>
      <button class="btn primary" data-act="reuse">Use this prompt and settings</button>
      <button class="btn" data-act="keepSeed">Keep this seed</button>
      <div class="actions-grid">
        <button class="btn" data-act="download" ${c.status === "done" ? "" : "disabled"}>Download</button>
        <button class="btn danger" data-act="deleteClip">Delete</button>
      </div>
    </div>`;
}

// An edited image's page: the picture it was made from.
function editedFrom(c) {
  const r = c.edit.id && state.refs.find((x) => x.id === c.edit.id);
  if (r) refSize(r);
  return `<div class="field">
      <span class="label">Edited picture</span>
      <div class="side-slots"><div class="side-slot" title="@${esc(c.edit.name)}">
        <span class="side-pic" style="background:${r ? r.tone : "transparent"}">${r ? `<img src="${r.img}" alt="">` : "?"}</span>
        <span class="slot-res">${r ? (r.w ? r.w + "×" + r.h : "") : esc(c.edit.name)}</span></div></div>
    </div>`;
}

// A Klein image's page: its prompt, the pictures it was made from (in their slots) and its settings.
function renderKleinSide(c) {
  const swap = !!c.swap, all = (c.slots || []).concat([null, null, null, null, null]);
  const slots = swap ? all.slice(0, Math.max(2, all.slice(0, 5).map(Boolean).lastIndexOf(true) + 1)) : all.slice(0, 4);
  const done = c.status === "done", name = (i) => (swap ? (i === 0 ? "body" : i === 1 ? "face" : "face " + i) : "image " + (i + 1));
  $("clipSide").innerHTML = `
    <div class="field">
      <span class="label">${swap ? "Face swap · notes" : "Prompt"}</span>
      <div class="prompt-box"><p class="clip-prompt">${swap ? (c.prompt === "Face swap" ? '<span class="label">None</span>' : esc(c.prompt)) : promptHTML(c.prompt, "klein", slots)}</p>
        <button class="copy prompt-copy" data-act="copyPrompt" aria-label="Copy the prompt" title="Copy the prompt">${ICON.copy}</button></div>
    </div>
    <div class="field">
      <span class="label">Pictures</span>
      <div class="side-slots">${slots.map((x, i) => {
        const r = x && state.refs.find((y) => y.id === x.id);
        if (r) refSize(r);
        return `<div class="side-slot ${x ? "" : "vacant"}" title="${x ? name(i) + ": @" + esc(x.name) : name(i) + " was empty"}">
          <span class="side-pic" style="background:${r ? r.tone : "transparent"}">${r ? `<img src="${r.img}" alt="">` : x ? "?" : ""}</span>
          <span class="side-name">${name(i)}</span><span class="slot-res">${r && r.w ? r.w + "×" + r.h : x ? "@" + esc(x.name) : "empty"}</span></div>`;
      }).join("")}</div>
    </div>
    ${swap && c.sent ? `<details class="sent-box"><summary>What the model got</summary><pre>${esc(c.sent)}</pre></details>` : ""}
    <dl class="facts">
      <dt>Model</dt><dd>${esc(c.model)}${swap ? " + " + esc(((OPT.klein.swapLoras || []).find((x) => x.file === c.bfs) || {}).name || "BFS head swap") : ""}</dd>
      ${swap ? `<dt>Head</dt><dd>${c.mask ? "Only the head changed" + (c.sampler === "lanpaint" ? " · LanPaint" : "") : "Whole picture"}${c.refine ? " · Refine face " + Math.round(c.refineStrength * 100) + "%" : ""}</dd>` : ""}
      <dt>Steps</dt><dd>${c.steps}</dd>
      <dt>Guidance</dt><dd>${Number(c.cfg).toFixed(1)}</dd>
      <dt>Size</dt><dd>${c.size && c.size !== "match" ? esc(c.size) : swap ? "Match the body" : "Match image 1"} · ${c.width}×${c.height} (${mpText(c.res)})</dd>
      <dt>Seed</dt><dd class="seed-dd">${c.seed}<button class="copy" data-act="copySeed" aria-label="Copy seed">${ICON.copy}</button></dd>
      <dt>LoRAs</dt><dd>${c.loras.length ? c.loras.map((l) => esc(loraText(l))).join("<br>") : '<span class="label">None</span>'}</dd>
    </dl>
    ${c.imported ? `<p class="hint imported">Imported from ${esc(c.imported)}${c.found === "graph" ? ". Settings worked out from its ComfyUI workflow" : ""}.</p>` : ""}
    <div class="actions">
      <button class="btn accent" data-act="editThis" ${done ? "" : "disabled"}>${ICON.klein}Edit this</button>
      <button class="btn" data-act="useInVideo" ${done ? "" : "disabled"}>${ICON.video}Use in a video</button>
      <button class="btn primary" data-act="reuse">Use this prompt and settings</button>
      <button class="btn" data-act="keepSeed">Keep this seed</button>
      <div class="actions-grid">
        <button class="btn" data-act="download" ${done ? "" : "disabled"}>Download</button>
        <button class="btn danger" data-act="deleteClip">Delete</button>
      </div>
    </div>`;
}

// A Pruna image's page: what you wrote (or the sheet's views), the pictures, what the model got (and with Enhance,
// what the upsampler wrote), and the settings.
function renderPrunaSide(c) {
  const sheet = c.mode === "sheet", done = c.status === "done";
  const count = sheet ? 4 : 9, list = (c.slots || []).concat(Array(count).fill(null)).slice(0, count);
  const roles = (c.sheet && c.sheet.roles) || ["character", "face", "outfit", "detail"];
  const name = (i) => (sheet ? (i ? roles[i] : "character") : "picture " + (i + 1));
  const shown = sheet ? list : list.slice(0, Math.max(1, list.map(Boolean).lastIndexOf(true) + 1));
  const views = sheet && c.sheet ? Object.keys(SHEET_VIEWS).filter((v) => c.sheet.views.includes(v)).map((v) => SHEET_VIEWS[v][0] + " " + SHEET_VIEWS[v][1].toLowerCase()) : [];
  $("clipSide").innerHTML = `
    <div class="field">
      <span class="label">${sheet ? "Notes" : "Prompt"}</span>
      <div class="prompt-box"><p class="clip-prompt">${sheet && c.prompt === "Character sheet" ? '<span class="label">None</span>' : promptHTML(c.prompt, sheet ? "prunaSheet" : "pruna", list.map((x) => x && x.id))}</p>
        <button class="copy prompt-copy" data-act="copyPrompt" aria-label="Copy the prompt" title="Copy the prompt">${ICON.copy}</button></div>
    </div>
    ${sheet ? `<div class="field"><span class="label">Views</span><p class="sheet-views-text">${views.map(esc).join(" · ")}</p></div>` : ""}
    ${list.some(Boolean) ? `<div class="field">
      <span class="label">Pictures</span>
      <div class="side-slots">${shown.map((x, i) => {
        const r = x && state.refs.find((y) => y.id === x.id);
        if (r) refSize(r);
        return `<div class="side-slot ${x ? "" : "vacant"}" title="${x ? name(i) + ": @" + esc(x.name) : name(i) + " was empty"}">
          <span class="side-pic" style="background:${r ? r.tone : "transparent"}">${r ? `<img src="${r.img}" alt="">` : x ? "?" : ""}</span>
          <span class="side-name">${esc(name(i))}</span><span class="slot-res">${r && r.w ? r.w + "×" + r.h : x ? "@" + esc(x.name) : "empty"}</span></div>`;
      }).join("")}</div>
    </div>` : ""}
    ${c.sent ? `<details class="sent-box" ${c.upsampled ? "" : "open"}><summary>${c.enhance ? "Your request, as the upsampler got it" : "What the model got"}</summary><pre>${esc(c.sent)}</pre></details>` : ""}
    ${c.enhance ? `<div class="field"><span class="label">What the upsampler wrote (what the model got)</span>
      ${c.upsampled ? `<pre class="upsampled">${esc(c.upsampled)}</pre>` : `<span class="hint">${isBusy(c) ? "Shows here when the picture is done." : "Not recorded."}</span>`}</div>` : ""}
    <dl class="facts">
      <dt>Model</dt><dd>${esc(c.model)}</dd>
      <dt>Kind</dt><dd>${sheet ? "Character sheet" : list.some(Boolean) ? "Edit of picture 1" : "From the prompt"}</dd>
      <dt>Enhance</dt><dd>${c.enhance ? "On" : "Off"}</dd>
      <dt>Steps</dt><dd>${c.steps}</dd>
      <dt>Size</dt><dd>${sheet ? "Sheet" : c.size && c.size !== "match" ? esc(c.size) : list[0] ? "Match picture 1" : "1:1"} · ${c.width}×${c.height} (${mpText(c.res)})</dd>
      <dt>Seed</dt><dd class="seed-dd">${c.seed}<button class="copy" data-act="copySeed" aria-label="Copy seed">${ICON.copy}</button></dd>
      <dt>LoRAs</dt><dd>${c.loras.length ? c.loras.map((l) => esc(loraText(l))).join("<br>") : '<span class="label">None</span>'}</dd>
    </dl>
    ${c.imported ? `<p class="hint imported">Imported from ${esc(c.imported)}.</p>` : ""}
    <div class="actions">
      <button class="btn accent" data-act="editThis" ${done ? "" : "disabled"}>${ICON.pruna}Edit this</button>
      <button class="btn" data-act="useInVideo" ${done ? "" : "disabled"}>${ICON.video}Use in a video</button>
      <button class="btn primary" data-act="reuse">Use this prompt and settings</button>
      ${c.upsampled ? `<button class="btn" data-act="useUpsampled">Use the upsampled prompt</button>` : ""}
      <button class="btn" data-act="keepSeed">Keep this seed</button>
      <div class="actions-grid">
        <button class="btn" data-act="download" ${done ? "" : "disabled"}>Download</button>
        <button class="btn danger" data-act="deleteClip">Delete</button>
      </div>
    </div>`;
}

// The player: a looping <video> with our own controls.
function wirePlayer() {
  const v = $("pv");
  if (!v) return;
  const c = clipById(state.openId);
  // Hold on to this player's own controls: the video can still fire events after the page has moved on.
  const btn = $("playBtn"), fill = $("playFill"), time = $("playTime");
  const update = () => {
    const d = v.duration || c.len;
    fill.style.width = (v.currentTime / d) * 100 + "%";
    time.textContent = clock(v.currentTime) + " / " + clock(d);
  };
  v.addEventListener("timeupdate", update);
  v.addEventListener("loadedmetadata", update);
  v.addEventListener("play", () => { btn.innerHTML = ICON.pause; btn.setAttribute("aria-label", "Pause"); player.classList.remove("paused"); });
  v.addEventListener("pause", () => { btn.innerHTML = ICON.play; btn.setAttribute("aria-label", "Play"); player.classList.add("paused"); });
  // The controls show while the mouse moves over the video, and fade out 2 s after it stops (or leaves).
  // They stay while paused or while a control has the keyboard focus.
  const player = v.closest(".player");
  let idle = null;
  const wake = () => { player.classList.add("awake"); clearTimeout(idle); idle = setTimeout(() => player.classList.remove("awake"), 2000); };
  player.addEventListener("pointermove", wake);
  player.addEventListener("pointerdown", wake);
  player.addEventListener("pointerleave", () => { clearTimeout(idle); player.classList.remove("awake"); });
  player.classList.toggle("paused", v.paused);
  // A click on the picture plays or pauses it (a double-click is full screen).
  v.addEventListener("click", () => { if (v.paused) v.play().catch(() => {}); else v.pause(); });
  $("scrub").addEventListener("pointerdown", (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    if (v.duration) v.currentTime = ((e.clientX - box.left) / box.width) * v.duration;
  });
}

function syncFullscreenButton() {
  const b = document.querySelector(".fs-btn");
  if (!b) return;
  b.innerHTML = isFull() ? ICON.shrink : ICON.expand;
  b.setAttribute("aria-label", isFull() ? "Exit full screen" : "Full screen");
}

/* ------------------------------------------------------------------
   Clips from the server: new ones, progress and finished files
   ------------------------------------------------------------------ */
function upsertClip(clip) {
  if (deleted.has(clip.id)) return;
  const old = clipById(clip.id);
  if (!old) {
    state.clips.unshift(clip);
    render();
    return;
  }
  const changed = old.status !== clip.status || old.file !== clip.file;
  // The server doesn't keep the live step or preview of a real render; those come straight from ComfyUI.
  const live = { step: old.step || 0, preview: old.preview };
  Object.assign(old, clip);
  if (isBusy(old)) { old.step = Math.max(live.step, clip.step || 0); old.preview = live.preview; }
  else {
    old.preview = null;
    if (live.preview && live.preview.startsWith("blob:")) setTimeout(() => URL.revokeObjectURL(live.preview), 3000);
  }
  if (changed) {
    if (state.view === "gallery" && state.tab === "edit") { renderHeader(); EDIT.clipsChanged(); }
    else if (state.view === "gallery") { renderHeader(); renderFilters(); renderGrid(); } else renderClip();
    return;
  }
  showProgress(old);
}

// Only the step moved: update the few things that show it, so nothing under the pointer is redrawn.
function showProgress(clip) {
  document.querySelectorAll(`[data-prog="${clip.id}"]`).forEach((el) => { el.style.width = barWidth(clip); });
  document.querySelectorAll(`[data-progtext="${clip.id}"]`).forEach((el) => { el.textContent = progressText(clip); });
  document.querySelectorAll(`[data-takeprog="${clip.id}"]`).forEach((el) => { el.textContent = takeProgress(clip); });
}

/* A real render's steps and its moving preview come from ComfyUI as it samples. */
const SAMPLER_NODE = "15";

function onProgress(d) {
  const clip = state.clips.find((c) => c.prompt_id && c.prompt_id === d.prompt_id);
  if (!clip || !isBusy(clip) || String(d.node) !== SAMPLER_NODE) return;
  const wasQueued = clip.status === "queued";
  clip.status = "rendering";
  clip.step = Math.max(clip.step || 0, Number(d.value) || 0);
  if (wasQueued) redrawClip(); else showProgress(clip);
}

// The live preview of a video: each step, a short clip of the whole video so far (MMVisionLivePreview).
function liveVideo(c) {
  return `<video data-live="${c.id}" src="${c.preview}" autoplay loop muted playsinline></video>`;
}

function onPreview(d) {
  const clip = state.clips.find((c) => c.prompt_id && c.prompt_id === d.prompt_id);
  if (!clip || !isBusy(clip) || !d.filename) return;
  const first = !clip.preview;
  clip.preview = fileUrl(d.subfolder, d.filename);
  clip.status = "rendering";
  if (d.step) clip.step = Math.max(clip.step || 0, Math.min(Number(d.step), clip.steps));
  if (first) { redrawClip(); return; }
  document.querySelectorAll(`[data-live="${clip.id}"]`).forEach((v) => { v.src = clip.preview; v.play?.().catch(() => {}); });
  showProgress(clip);
}

// A live preview picture for any render (an image's preview arrives as a picture, not a file).
function setLivePreview(clip, url, step) {
  if (!clip || !isBusy(clip)) return;
  const first = !clip.preview;
  clip.preview = url;
  clip.status = "rendering";
  if (step) clip.step = Math.max(clip.step || 0, Math.min(step, clip.steps));
  if (first) { redrawClip(); return; }
  document.querySelectorAll(`[data-live="${clip.id}"]`).forEach((el) => { el.src = url; });
  showProgress(clip);
}

// ComfyUI's live preview of an image while it samples: a small JPEG, sent as a binary message.
// Type 4 carries the job's prompt id; the older type 1 doesn't, so it goes to the image that is rendering.
const PREVIEW_IMAGE = 1, PREVIEW_IMAGE_WITH_METADATA = 4;

function onPreviewPicture(buf) {
  if (buf.byteLength < 8) return;
  const view = new DataView(buf);
  const event = view.getUint32(0);
  let clip, bytes, type = "image/jpeg";
  if (event === PREVIEW_IMAGE_WITH_METADATA) {
    const size = view.getUint32(4);
    let meta = {};
    try { meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, size))); } catch { return; }
    clip = state.clips.find((c) => c.prompt_id && c.prompt_id === meta.prompt_id);
    bytes = buf.slice(8 + size);
    type = meta.image_type || type;
  } else if (event === PREVIEW_IMAGE) {
    clip = state.clips.find((c) => kindOf(c) === "image" && c.status === "rendering" && c.prompt_id);
    bytes = buf.slice(8);
    if (view.getUint32(4) === 2) type = "image/png";
  } else return;
  // Videos have their own moving preview (mmvision_preview); these pictures are for images only.
  if (!clip || kindOf(clip) !== "image" || !isBusy(clip)) return;
  const old = clip.preview;
  setLivePreview(clip, URL.createObjectURL(new Blob([bytes], { type })));
  if (old && old.startsWith("blob:")) setTimeout(() => URL.revokeObjectURL(old), 3000);
}

function redrawClip() {
  if (state.view === "gallery") { renderHeader(); renderGrid(); } else renderClip();
}

function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws?clientId=mmflow-" + Math.random().toString(36).slice(2));
  ws.binaryType = "arraybuffer";
  // Ask ComfyUI to say which job each preview picture belongs to.
  ws.addEventListener("open", () => ws.send(JSON.stringify({ type: "feature_flags", data: { supports_preview_metadata: true } })));
  ws.addEventListener("message", (e) => {
    if (e.data instanceof ArrayBuffer) { onPreviewPicture(e.data); return; }
    if (typeof e.data !== "string") return;
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.type === "mmflow.clip" && msg.data?.clip) upsertClip(msg.data.clip);
    if (msg.type === "mmflow.loras") { if (LM.open) { XFER.fresh.add(msg.data?.file); pollLinks(); fetchLoras(false); } else api("GET", "loras").then((r) => { OPT.loras = r.loras; LM.sizes = r.sizes || {}; }).catch(() => {}); }
    if (msg.type === "mmflow.prompts" && typeof PB !== "undefined" && PB.open && !PB.dirty) pbPollDrive();
    if (msg.type === "mmflow.refmods") {
      const j = msg.data?.job || {};
      loadRefmods().then(() => {
        if (j.state === "done") {
          const s = state.videoSettings;
          if (!(s.refmods || []).some((m) => m.file === j.name)) s.refmods = [...(s.refmods || []), { file: j.name, strength: 1 }].slice(0, 8);
          saveUi(); if (state.tab === "video") render();
          say(`The RefMod ${j.name} is ready, and added to the Video settings.`);
        } else if (j.state === "failed") say(`Couldn't make the RefMod ${j.name}: ${j.error}`);
        else if (j.state === "fetched") { if (state.tab === "video" && state.settingsOpen) render(); say("RefMods from Google Drive are here."); }
      });
    }
    if (msg.type === "mmflow.models") { load(false); if (MODELS.open) pollModels(false); }
    else if (msg.type === "progress" && msg.data) onProgress(msg.data);
    else if (msg.type === "mmvision_preview" && msg.data) onPreview(msg.data);
    else if (msg.type === "mmvision_export" && msg.data) EDIT.onExport(msg.data);
  });
  // ComfyUI restarted or the connection dropped: reconnect, and reload the state in case something finished meanwhile.
  ws.addEventListener("close", () => setTimeout(() => { connect(); load(false); }, 2000));
}

// What Generate sends for the tab as it is now: {body}, or {error} (said instead). The "What H3 gets" preview
// asks the server with the same body, so it shows exactly what a render would send.
function generateBody() {
  const s = state.settings;
  const prompt = state.prompt.trim();
  if (isPruna()) {
    const sheet = sheetOn(), list = slots();
    if (sheet && !list[0]) return { error: "Add the character picture first." };
    if (!sheet && !prompt) return { error: "Write a prompt first." };
    const empty = emptyNamed(prompt);
    if (empty.length) return { error: `The prompt names ${empty.map((n) => "picture " + n).join(" and ")}, but ${empty.length > 1 ? "those slots are" : "that slot is"} empty. Fill it, or change the prompt.` };
    return { body: { kind: "pruna", mode: state.prunaMode, prompt, settings: state.prunaSettings, slots: list,
      sheet: sheet ? { text: sheetText(), ratio: sheetLayout().ratio, ...state.prunaSheet } : undefined } };
  }
  if (swapOn()) {
    const [body, face] = state.kleinSwapSlots;
    if (!OPT.klein.swap) return { error: "Face swap needs the BFS head-swap LoRA for Klein 9B in models/loras." };
    if (!body || !face) return { error: body ? "Add the face picture too." : "Add the body picture first, then the face." };
    return { body: { kind: "klein", mode: "swap", prompt, settings: s, slots: state.kleinSwapSlots,
      mask: headOnly() ? state.kleinSwapMask.mask : undefined } };
  }
  const picked = state.tab === "video" && state.motion && motionSource(state.motion) && state.motionPeople.length;
  if (!prompt && !picked) return { error: "Write a prompt first." };
  const names = new Set(state.refs.map((r) => r.name));
  const unknown = imageTab() || state.tab === "video" ? null : mentionsIn(prompt).find((n) => !names.has(n));
  if (unknown) return { error: "There is no image called @" + unknown + "." };
  if (editOn() && !state.imageEditSlot[0]) return { error: "Add the picture to edit first (or switch Edit off)." };
  const empty = isKlein() ? emptyNamed(prompt) : [];
  if (empty.length) return { error: `The prompt names ${empty.map((n) => "image " + n).join(" and ")}, but ${empty.length > 1 ? "those slots are" : "that slot is"} empty. Fill it, or change the prompt.` };
  return { body: { prompt, settings: s, kind: state.tab, negative: isImage() ? state.negative : undefined,
    slots: isKlein() ? state.kleinSlots : undefined, edit: editOn() ? state.imageEditSlot[0] : undefined,
    pictures: state.tab === "video" ? state.attached.filter((id) => state.refs.some((r) => r.id === id)) : undefined,
    motion: state.tab === "video" && state.motion && motionSource(state.motion) ? state.motion : undefined,
    motionMethod: motionMethodNow(),
    motionPeople: state.tab === "video" && state.motion && motionSource(state.motion) && state.motionPeople.length ? state.motionPeople : undefined,
    motionExtra: state.motionPeople.length && state.motionExtra.trim() ? state.motionExtra.trim() : undefined } };
}

async function generate() {
  if (!state.ready) { say("MiniMax H3 Vision hasn't loaded from the server yet."); return; }
  const g = generateBody();
  if (g.error) { say(g.error); return; }
  state.settingsOpen = false;
  if (!isPruna() && !swapOn()) state.filter = "all";
  render();
  try {
    const { clip } = await api("POST", "generate", g.body);
    upsertClip(clip);
  } catch (error) {
    say(error.message);
  }
}

// ---- What the model gets: hover the eye in the prompt bar ----
{
  const wrap = () => document.querySelector(".sent-wrap");
  document.addEventListener("mouseover", (e) => {
    if (!wrap()?.contains(e.target)) return;
    clearTimeout(SENT.timer);
    SENT.timer = setTimeout(() => showSent(), 250);
  });
  document.addEventListener("mouseout", (e) => {
    if (!wrap()?.contains(e.target) || wrap().contains(e.relatedTarget)) return;
    clearTimeout(SENT.timer);
    SENT.timer = setTimeout(() => hideSent(), 300);
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("sentBox").hidden) hideSent(true); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && state.pickingMotion) { state.pickingMotion = false; render(); } });
}
const SENT = { timer: null, key: "", pinned: false, busy: false };
function sentKey() { try { return JSON.stringify(generateBody()); } catch { return ""; } }
async function showSent(pin) {
  const box = $("sentBox");
  if (pin !== undefined) SENT.pinned = pin;
  box.hidden = false;
  $("sentBtn").setAttribute("aria-expanded", "true");
  const key = sentKey();
  if (key === SENT.key && !box.classList.contains("stale")) return;
  SENT.key = key;
  const g = generateBody();
  if (g.error) { box.innerHTML = sentHTML(null, g.error); return; }
  box.classList.add("stale");
  try {
    const r = await api("POST", "preview", g.body);
    if (SENT.key !== key) return;
    box.innerHTML = sentHTML(r);
  } catch (error) { box.innerHTML = sentHTML(null, error.message); }
  box.classList.remove("stale");
}
function hideSent(force) {
  if (SENT.pinned && !force) return;
  SENT.pinned = false;
  $("sentBox").hidden = true;
  $("sentBtn").setAttribute("aria-expanded", "false");
}
function sentHTML(r, error) {
  if (!r) return `<div class="sent-head"><b>What the model gets</b></div><p class="sent-note">${esc(error)}</p>`;
  const parts = r.texts.map((t) => `<div class="sent-part"><span class="sent-label">${esc(t.label)}</span><pre class="sent-text">${esc(t.text) || "<i>(empty)</i>"}</pre></div>`).join("");
  const mods = (r.refmods || []).length ? `<div class="sent-part"><span class="sent-label">RefMods</span><ul class="sent-pics">${r.refmods.map((m) => `<li>${esc(m)}</li>`).join("")}</ul></div>` : "";
  const pics = r.pictures.length ? `<div class="sent-part"><span class="sent-label">Pictures, in order</span><ul class="sent-pics">${r.pictures.map((p, i) => `<li><b>Picture ${i + 1}</b> ${esc(p)}</li>`).join("")}</ul></div>` : "";
  return `<div class="sent-head"><b>What ${esc(r.model || "the model")} gets</b><button class="link" data-act="copySent">Copy</button></div>${parts}${pics}${mods}${(r.files || []).length ? `<div class="sent-part"><span class="sent-label">Model file</span><ul class="sent-pics">${r.files.map((f) => `<li>${esc(f)}</li>`).join("")}</ul></div>` : ""}
    <p class="sent-note">${SENT.pinned ? "Click the eye again to close." : "Click the eye to keep this open."}</p>`;
}

/* ------------------------------------------------------------------
   Hover preview in the grid: the clip plays and loops while the pointer is over it,
   and goes back to its first frame when the pointer leaves.
   ------------------------------------------------------------------ */
const hover = { id: null };

function hoverStart(id) {
  const c = clipById(id);
  if (!c || !c.file || kindOf(c) === "image" || hover.id === id) return;
  hoverStop();
  hover.id = id;
  requestAnimationFrame(hoverFrame);
}

function hoverStop() {
  if (!hover.id) return;
  const id = hover.id;
  hover.id = null;
  const tile = document.querySelector(`.tile[data-clip="${id}"]`);
  if (!tile) return;
  const v = tile.querySelector("video");
  if (v) { v.pause(); v.currentTime = 0; }
  const bar = tile.querySelector("[data-hoverbar]");
  if (bar) bar.style.width = "0";
  const c = clipById(id);
  if (c) tile.querySelector("[data-len]").textContent = c.len + " s";
  tile.classList.remove("hovering");
}

function hoverFrame() {
  if (!hover.id) return;
  // Look the tile up each frame, so a redraw of the grid doesn't stop the preview.
  const tile = document.querySelector(`.tile[data-clip="${hover.id}"]`);
  const v = tile?.querySelector("video");
  if (!v) { hover.id = null; return; }
  tile.classList.add("hovering");
  if (v.paused) v.play().catch(() => {});
  const d = v.duration || clipById(hover.id).len;
  tile.querySelector("[data-hoverbar]").style.width = (v.currentTime / d) * 100 + "%";
  tile.querySelector("[data-len]").textContent = clock(v.currentTime) + " / " + clock(d);
  requestAnimationFrame(hoverFrame);
}

document.addEventListener("mouseover", (e) => {
  const tile = e.target.closest(".tile[data-clip]");
  if (tile) hoverStart(tile.dataset.clip);
  else hoverStop();
});
document.addEventListener("mouseleave", hoverStop);

// Text to the clipboard. Without https (no navigator.clipboard), the old way: a hidden text box and copy.
async function copyText(text, done) {
  try { await navigator.clipboard.writeText(text); say(done); return; } catch {}
  const t = document.createElement("textarea");
  t.value = text; t.style.cssText = "position:fixed;left:-9999px;top:0";
  document.body.appendChild(t); t.select();
  const ok = document.execCommand("copy");
  t.remove();
  say(ok ? done : "Couldn't copy here: select the prompt on the clip's page and copy it.");
}

/* ------------------------------------------------------------------
   Right-click menu on a clip
   ------------------------------------------------------------------ */
let menuFor = null;

function openMenu(id, x, y) {
  const c = clipById(id);
  if (!c) return;
  menuFor = id;
  const m = $("ctx");
  m.innerHTML = `
    <button role="menuitem" data-act="menuDownload" ${c.status === "done" ? "" : 'aria-disabled="true" class="off"'}>${ICON.download}Download</button>${kindOf(c) === "video" ? `
    <button role="menuitem" data-act="menuExportZip" ${c.status === "done" && c.file ? "" : 'aria-disabled="true" class="off"'}>${ICON.download}Export .zip with reference images</button>${DRIVE.on ? `
    <button role="menuitem" data-act="menuDriveZip" ${c.status === "done" && c.file ? "" : 'aria-disabled="true" class="off"'}>${ICON.upload}Save .zip to Google Drive</button>` : ""}` : ""}
    <button role="menuitem" data-act="menuDelete" class="danger">${ICON.trash}${isBusy(c) ? "Cancel and delete" : "Delete"}</button>
    <span class="ctx-line"></span>${kindOf(c) === "video" ? (EDIT.has(c.id)
      ? `<button role="menuitem" aria-disabled="true" class="off" title="This video is on the edit's timeline already.">${ICON.check}Already in the edit</button>`
      : `<button role="menuitem" data-act="menuAddToEdit" ${c.status === "done" && c.file ? "" : 'aria-disabled="true" class="off"'}>${ICON.edit}Add to edit</button>`) : ""}${kindOf(c) === "video" ? `
    <button role="menuitem" data-act="menuUseMotion" ${c.status === "done" && c.file ? "" : 'aria-disabled="true" class="off"'} title="Make a new video with this one's motion: your pictures and prompt say who performs it">${ICON.swap}Use its motion</button>
    <button role="menuitem" data-act="menuSwap" ${c.status === "done" && c.file ? "" : 'aria-disabled="true" class="off"'} title="Put the person from a picture into this video, with its motion and sound">${ICON.swap}Swap character…</button>
    <button role="menuitem" data-act="menuRefmod" ${c.status === "done" && c.file && RM.installed ? "" : 'aria-disabled="true" class="off"'} title="${RM.installed ? "Save this video's style, look or motion as a RefMod, to use in other videos" : "RefMods need the ComfyUI-MiniMaxH3Mod node pack: on RunPod, set ENABLE_REFMOD=true."}">${ICON.plus}Make a RefMod from it…</button>
    <button role="menuitem" data-act="menuFaces" ${c.status === "done" && c.file ? "" : 'aria-disabled="true" class="off"'} title="${esc(FACES.why || "Re-render small faces up close and put them back, as a new clip")}">${ICON.face}Refine faces…</button>` : ""}
${plainVideo(c) ? "" : `
    <button role="menuitem" data-act="menuReuse">${ICON.reuse}Use this prompt and settings</button>`}
    <button role="menuitem" data-act="menuCopyPrompt" ${c.prompt ? "" : 'aria-disabled="true" class="off"'}>${ICON.copy}Copy prompt</button>${kindOf(c) === "image" ? `
    <button role="menuitem" data-act="menuUseInVideo" ${c.status === "done" ? "" : 'aria-disabled="true" class="off"'}>${ICON.video}Use in a video</button>
    <button role="menuitem" data-act="menuEditThis" ${c.status === "done" ? "" : 'aria-disabled="true" class="off"'}>${tabOf(c) === "pruna" ? ICON.pruna + "Edit this (picture 1)" : ICON.klein + (tabOf(c) === "klein" ? "Edit this (image 1)" : "Edit in Klein (image 1)")}</button>${tabOf(c) === "pruna" ? "" : `
    <button role="menuitem" data-act="menuEditInPruna" ${c.status === "done" ? "" : 'aria-disabled="true" class="off"'}>${ICON.pruna}Edit in Pruna (picture 1)</button>`}` : ""}`;
  m.hidden = false;
  // Keep the whole menu inside the window.
  const r = m.getBoundingClientRect();
  m.style.left = Math.min(x, window.innerWidth - r.width - 8) + "px";
  m.style.top = Math.min(y, window.innerHeight - r.height - 8) + "px";
  m.querySelector("button:not(.off)").focus();
}

function closeMenu() {
  const id = menuFor;
  menuFor = null;
  $("ctx").hidden = true;
  return id;
}

document.addEventListener("contextmenu", (e) => {
  const host = e.target.closest("[data-clip]");
  if (!host) { closeMenu(); return; }
  e.preventDefault();
  // From the keyboard (menu key or Shift+F10) there is no pointer, so open at the clip.
  const keyboard = e.clientX === 0 && e.clientY === 0;
  const box = host.getBoundingClientRect();
  openMenu(host.dataset.clip, keyboard ? box.left + 16 : e.clientX, keyboard ? box.top + 16 : e.clientY);
});
document.addEventListener("pointerdown", (e) => { if (menuFor && !e.target.closest("#ctx")) closeMenu(); });
window.addEventListener("scroll", () => menuFor && closeMenu(), true);
window.addEventListener("blur", () => menuFor && closeMenu());
$("ctx").addEventListener("keydown", (e) => {
  const items = [...$("ctx").querySelectorAll("button:not(.off)")];
  const i = items.indexOf(document.activeElement);
  if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
  if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
  if (e.key === "Escape" || e.key === "Tab") { e.preventDefault(); closeMenu(); }
});

/* ------------------------------------------------------------------
   Actions. Buttons name one with data-act; inputs with data-change or data-input.
   ------------------------------------------------------------------ */
function openClip(id) {
  hoverStop();
  state.view = "clip";
  state.openId = id;
  state.settingsOpen = false;
  state.library.open = false;
  shownClip = null;
  render();
  window.scrollTo(0, 0);
}

async function removeClip(id, message) {
  deleted.add(id);
  state.clips = state.clips.filter((x) => x.id !== id);
  if (state.openId === id) { state.view = "gallery"; state.openId = null; }
  render();
  try {
    await api("DELETE", "clips/" + id);
    say(message);
  } catch (error) {
    say("Couldn't delete: " + error.message);
  }
}

/* ------------------------------------------------------------------
   Add LoRA: a window listing the LoRA files in ComfyUI's loras folders
   ------------------------------------------------------------------ */
const LM = { open: false, q: "", sel: 0, sizes: {}, busy: false, all: false };
// LoRAs are sorted by model in folders inside models/loras: "MiniMax H3" for the Video tab, "MOSS Image" for the
// Image tab, "Flux Klein" for Klein. Each tab lists its own folder and the loose files in loras itself
// ("Not sorted"); Show all lists every folder. Folder names are matched loosely (minimax_h3, moss, klein…).
const LORA_FOLDER = { video: "MiniMax H3", image: "MOSS Image", klein: "Flux Klein", pruna: "P-H3 Image" };
const loraTop = (n) => { const p = n.replace(/\\/g, "/"); return p.includes("/") ? p.slice(0, p.indexOf("/")) : ""; };
function loraFamily(n) {
  const top = loraTop(n).toLowerCase().replace(/[^a-z0-9]/g, "");
  if (!top) return "";
  if (top.startsWith("ph3") || top.startsWith("pruna")) return "pruna";
  if (top.startsWith("minimax") || top.startsWith("h3") || top === "hyperflow") return "video";
  if (top.startsWith("moss")) return "image";
  if (top.includes("klein") || top.startsWith("flux")) return "klein";
  return "other";
}
const loraTab = () => (isImage() ? "image" : isKlein() ? "klein" : isPruna() ? "pruna" : "video");
const loraBase = (n) => n.replace(/\\/g, "/").split("/").pop();
const loraFolder = (n) => (n.includes("/") ? n.slice(0, n.lastIndexOf("/")) : n.includes("\\") ? n.slice(0, n.lastIndexOf("\\")) : "");
const loraStem = (n) => n.slice(Math.max(n.lastIndexOf("/"), n.lastIndexOf("\\")) + 1).replace(/\.[^.]+$/, "");
const loraSize = (b) => (b >= 1 << 30 ? (b / (1 << 30)).toFixed(1) + " GB" : b >= 1 << 20 ? Math.round(b / (1 << 20)) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");
// The LoRAs to list, in groups: this tab's folder, then Not sorted, then (with Show all) other models' folders.
// A folder that isn't one of the model folders counts as Not sorted too: only another model's LoRAs are hidden.
// Pruna's P-H3-Image runs on MiniMax H3, so the Pruna tab lists MiniMax H3's LoRAs as its own too.
const loraOwn = (f) => f === loraTab() || (loraTab() === "pruna" && f === "video");
const loraRank = (n) => { const f = loraFamily(n); return loraOwn(f) ? 0 : f === "" || f === "other" ? 1 : 2; };
const loraShown = () => {
  const words = LM.q.toLowerCase().split(/\s+/).filter(Boolean);
  return OPT.loras
    .filter((n) => LM.all || loraRank(n) < 2)
    // P-H3-Image's own LoRA is always on in the Pruna tab: it isn't one to add.
    .filter((n) => !/^p-h3-image_lora/i.test(loraBase(n)))
    .filter((n) => words.every((w) => n.toLowerCase().includes(w)))
    .sort((a, b) => loraRank(a) - loraRank(b) || XFER.fresh.has(b) - XFER.fresh.has(a) || a.toLowerCase().localeCompare(b.toLowerCase()));
};
// A LoRA in the settings counts as this file when the name is the same, or the file name is (it was moved).
const loraUsed = (n) => state.settings.loras.some((l) => l.id === n || loraBase(l.id) === loraBase(n));
function loraHeading(n) {
  const rank = loraRank(n), parts = n.replace(/\\/g, "/").split("/").slice(0, -1);
  if (rank === 1) return parts.length ? parts.join(" / ") + " · not sorted" : "Not sorted · loose in the loras folder";
  if (rank === 0) return parts.join(" / ");
  return parts.join(" / ") + " · for another model";
}

function openLoras() {
  LM.open = true; LM.q = ""; LM.sel = 0; LM.all = false;
  XFER.dest = null; XFER.fresh.clear();
  $("loraModal").hidden = false;
  renderLoraDest();
  renderLoras(true);
  renderXfer();
  pollLinks();
  $("loraQuery").focus();
  fetchLoras(false);
}
function closeLoras() {
  if (!LM.open) return;
  LM.open = false;
  $("loraModal").hidden = true;
  render();
  document.querySelector('[data-act="loraPicker"]')?.focus();
}
async function fetchLoras(refresh) {
  if (LM.busy) return;
  LM.busy = true; renderLoraHead();
  const before = new Set(OPT.loras);
  try {
    const r = await api("GET", "loras" + (refresh ? "?refresh=1" : ""));
    OPT.loras = r.loras; LM.sizes = r.sizes || {};
    if (refresh) {
      const added = r.loras.filter((n) => !before.has(n)).length, gone = [...before].filter((n) => !r.loras.includes(n)).length;
      say(`Found ${r.loras.length} LoRA${r.loras.length === 1 ? "" : "s"}` + (added ? `, ${added} new` : "") + (gone ? `, ${gone} gone` : "") + (added || gone ? "." : ". Nothing changed."));
    }
  } catch (error) {
    if (refresh) say("Couldn't look for LoRAs: " + error.message);
  }
  LM.busy = false;
  if (LM.open) {
    renderLoras();
    if (!$("loraModal").contains(document.activeElement)) (refresh ? $("loraRefresh") : $("loraQuery")).focus();
  }
}
function toggleLoraFile(name) {
  if (loraUsed(name)) state.settings.loras = state.settings.loras.filter((l) => l.id !== name && loraBase(l.id) !== loraBase(name));
  else state.settings.loras.push({ id: name, name, strength: 1, on: true });
  const list = loraShown();
  LM.sel = Math.max(0, list.indexOf(name));
  render();
  renderLoras();
  if (!$("loraModal").contains(document.activeElement)) $("loraQuery").focus();
}
function renderLoraHead() {
  const b = $("loraRefresh");
  if (!b) return;
  b.disabled = LM.busy;
  b.classList.toggle("spin", LM.busy);
  b.lastChild.textContent = LM.busy ? "Looking…" : "Refresh";
}
function renderLoras(first) {
  if (first) $("loraQuery").value = "";
  renderLoraHead();
  const list = loraShown();
  LM.sel = Math.min(LM.sel, Math.max(0, list.length - 1));
  const n = state.settings.loras.length, folder = LORA_FOLDER[loraTab()];
  const own = OPT.loras.filter((x) => loraRank(x) === 0).length, others = OPT.loras.filter((x) => loraRank(x) === 2).length;
  $("loraTitle").textContent = "Add a LoRA · " + folder;
  $("loraCount").textContent = `${own} in loras/${folder} · ${n} in use`;
  $("loraAll").checked = LM.all;
  $("loraAllText").textContent = `Show all LoRAs${others ? ` (${others} for other models)` : ""}`;
  const tip = `<p class="lm-empty">Put this tab's LoRAs in <code>models/loras/${esc(folder)}</code> and click Refresh.</p>`;
  if (!OPT.loras.length) {
    $("loraList").innerHTML = `<p class="lm-empty">No LoRAs in ComfyUI's loras folder yet.</p>` + tip;
    return;
  }
  if (!list.length) {
    $("loraList").innerHTML = LM.q ? `<p class="lm-empty">No LoRA matches “${esc(LM.q)}”.</p>` : `<p class="lm-empty">No LoRAs for ${esc(folder)} yet.</p>` + tip;
    return;
  }
  let heading = null, html = "";
  list.forEach((name, i) => {
    const h = loraHeading(name);
    if (h !== heading) { heading = h; if (h) html += `<div class="lm-folder ${loraRank(name) ? "other" : ""}">${esc(h)}</div>`; }
    const on = loraUsed(name);
    html += `<button class="lm-item ${on ? "on" : ""} ${i === LM.sel ? "sel" : ""} ${XFER.fresh.has(name) ? "fresh" : ""}" role="option" aria-selected="${i === LM.sel}" data-act="pickLora" data-id="${esc(name)}" title="${esc(name)}">
      <span class="lm-name">${esc(loraStem(name))}${XFER.fresh.has(name) ? ' <em class="lm-newtag">new</em>' : ""}</span>
      <span class="lm-size">${LM.sizes[name] ? loraSize(LM.sizes[name]) : ""}</span>
      <span class="lm-state">${on ? ICON.check + "Added" : ICON.plus + "Add"}</span></button>`;
  });
  $("loraList").innerHTML = html;
  $("loraList").querySelector(".lm-item.sel")?.scrollIntoView({ block: "nearest" });
}
$("loraQuery").addEventListener("input", (e) => { LM.q = e.target.value; LM.sel = 0; renderLoras(); });
$("loraAll").addEventListener("change", (e) => { LM.all = e.target.checked; LM.sel = 0; renderLoras(); $("loraQuery").focus(); });
$("loraModal").addEventListener("mousedown", (e) => { if (e.target === $("loraModal")) closeLoras(); });

// ---- Google Drive: project and video zips saved to and opened from a folder (GDRIVE_FOLDER on the pod) ----
// The pod does the transfers, so they go on with the window closed; when an open finishes, its videos are added
// wherever you are (and in Edit, a project's timeline is offered).
const DRIVE = { on: false, problem: null, jobs: [], files: null, error: null, open: false, timer: null, handled: new Set(), loading: false };
async function driveStatus() {
  try {
    const r = await api("GET", "gdrive");
    DRIVE.on = r.on; DRIVE.problem = r.problem;
    driveJobs(r.jobs, true);
  } catch { DRIVE.on = false; }
  $("driveItem").hidden = !DRIVE.on;
  EDIT.driveChanged();
}
function driveJobs(jobs, first) {
  const before = new Map(DRIVE.jobs.map((j) => [j.id, j.state]));
  DRIVE.jobs = jobs;
  for (const j of jobs) {
    if (first) { DRIVE.handled.add(j.id); continue; }
    if (DRIVE.handled.has(j.id) || !["done", "failed"].includes(j.state)) continue;
    DRIVE.handled.add(j.id);
    if (j.state === "failed") { say((j.kind === "save" ? "Couldn't save to Google Drive: " : "Couldn't open " + j.name + ": ") + j.error); continue; }
    if (j.kind === "save") { say(`Saved ${j.name} to Google Drive.`); DRIVE.files = null; if (DRIVE.open) driveFiles(true); }
    else driveOpened(j);
  }
  if (jobs.some((j) => !["done", "failed", "cancelled"].includes(j.state))) {
    clearTimeout(DRIVE.timer);
    DRIVE.timer = setTimeout(drivePoll, DRIVE.open ? 700 : 1500);
  }
  renderDrive();
  return before;
}
async function drivePoll() {
  try { driveJobs((await api("GET", "gdrive")).jobs); } catch { DRIVE.timer = setTimeout(drivePoll, 3000); }
}
async function driveOpened(job) {
  const r = job.result;
  if (!r) return;
  if (state.tab === "edit") { if (r.edit) closeDrive(); return EDIT.opened(r, job.name); }
  importedZip(r);
  if (r.clips.length === 1) { if (state.tab !== "video") ACT.tab("video"); openClip(r.clips[0].id); }
  else { if (state.tab !== "video") ACT.tab("video"); render(); }
  say(`Opened ${job.name}: ` + zipSummary(r) + "." + (r.edit ? " Its timeline can be opened from Edit › Project › Open from Google Drive." : ""));
}
async function driveSave(body) {
  try {
    const r = await api("POST", "gdrive/save", body);
    say("Saving to Google Drive…");
    driveJobs(r.jobs);
  } catch (error) { say(error.message); }
}
async function driveFiles(fresh) {
  DRIVE.loading = true; renderDrive();
  try { DRIVE.files = (await api("GET", "gdrive/files" + (fresh ? "?fresh=1" : ""))).files; DRIVE.error = null; }
  catch (error) { DRIVE.error = error.message; }
  DRIVE.loading = false; renderDrive();
}
function openDrive() {
  DRIVE.open = true;
  $("driveModal").hidden = false;
  renderDrive();
  driveFiles(true);
  drivePoll();
  setTimeout(() => $("driveModal").querySelector(".dm-file, .lm-close")?.focus(), 50);
}
function closeDrive() { DRIVE.open = false; $("driveModal").hidden = true; }
function driveRow(j) {
  const going = !["done", "failed", "cancelled"].includes(j.state);
  const pct = j.total ? Math.min(100, (100 * j.done) / j.total) : 0;
  const left = j.speed > 0 && j.total ? timeText((j.total - j.done) / j.speed) + " left" : "";
  const amount = [j.total ? `${bytesText(j.done)} of ${bytesText(j.total)}` : j.done ? bytesText(j.done) : "", j.speed > 0 ? bytesText(j.speed) + "/s" : "", left].filter(Boolean).join(" · ");
  const line = {
    queued: "Waiting for the one before", packing: "Making the .zip…", sending: amount || "Sending…", fetching: amount || "Fetching…",
    opening: "Adding its videos…", done: j.kind === "save" ? "Saved to Google Drive" : "Opened", failed: j.error, cancelled: "Cancelled.",
  }[j.state] || "";
  const name = j.name || (j.kind === "save" ? "Saving to Google Drive" : "");
  const btn = going && j.state !== "opening" ? `<button data-act="driveCancel" data-id="${esc(j.id)}">Cancel</button>` : "";
  const bar = going ? `<div class="lm-bar ${!j.total || ["packing", "queued", "opening"].includes(j.state) ? "wait" : ""}"><i style="width:${pct.toFixed(1)}%"></i></div>` : "";
  return `<div class="lm-x ${j.state}"><b>${j.kind === "save" ? "↑ " : "↓ "}${esc(name)}</b>${btn}<small>${esc(line)}</small>${bar}</div>`;
}
function renderDrive() {
  if (!DRIVE.open) return;
  const recent = DRIVE.jobs.filter((j) => !["done", "cancelled"].includes(j.state) || Date.now() / 1000 - j.at < 120);
  $("driveJobs").innerHTML = [...recent].reverse().map(driveRow).join("");
  const busy = new Set(DRIVE.jobs.filter((j) => j.kind === "open" && !["done", "failed", "cancelled"].includes(j.state)).map((j) => j.name));
  let html;
  if (DRIVE.problem) html = `<div class="dm-empty bad">${esc(DRIVE.problem)}</div>`;
  else if (DRIVE.error) html = `<div class="dm-empty bad">${esc(DRIVE.error)}</div>`;
  else if (!DRIVE.files) html = `<div class="dm-empty">Reading the folder…</div>`;
  else if (!DRIVE.files.length) html = `<div class="dm-empty">No .zip files in the folder yet.<br>Save one from Edit › Project, or a video's ⋯ menu.</div>`;
  else html = DRIVE.files.map((f) => `<button class="dm-file" data-act="driveOpen" data-id="${esc(f.name)}" ${busy.has(f.name) ? "disabled" : ""}>
      <b title="${esc(f.name)}">${esc(f.name.replace(/\.zip$/i, ""))}</b><small>${bytesText(f.size)} · ${esc(driveDate(f.modified))}</small><span>${busy.has(f.name) ? "Opening…" : "Open"}</span></button>`).join("");
  $("driveList").innerHTML = html;
  $("driveCount").textContent = DRIVE.files && !DRIVE.problem ? `${DRIVE.files.length} .zip file${DRIVE.files.length === 1 ? "" : "s"} in your folder` : "Your projects folder";
  $("driveModal").querySelector(".lm-refresh").classList.toggle("spin", DRIVE.loading);
}
function driveDate(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const same = d.toDateString() === new Date().toDateString();
  return same ? "today " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString([], { day: "numeric", month: "short", year: d.getFullYear() === new Date().getFullYear() ? undefined : "numeric" });
}
$("driveModal").addEventListener("mousedown", (e) => { if (e.target === $("driveModal")) closeDrive(); });
document.addEventListener("keydown", (e) => { if (DRIVE.open && e.key === "Escape" && $("confirm").hidden) { e.preventDefault(); e.stopPropagation(); closeDrive(); } }, true);

// ---- New LoRAs: uploaded from this computer in pieces, or fetched by the pod from a link ----
// An upload sends 8 MB at a time; a piece that fails is sent again (the server says where it is), so a flaky
// connection slows it down rather than losing it. Links are fetched on the pod, so the window can be closed.
const XFER = { ups: [], links: [], dest: null, timer: null, fresh: new Set() };
const UP_PIECE = 8 * 1024 * 1024;
const LORA_DESTS = [["MiniMax H3", "video"], ["P-H3 Image", "pruna"], ["MOSS Image", "image"], ["Flux Klein", "klein"], ["", ""]];
function renderLoraDest() {
  const sel = $("loraDest");
  if (XFER.dest === null) XFER.dest = LORA_FOLDER[loraTab()];
  sel.innerHTML = LORA_DESTS.map(([f]) => `<option value="${esc(f)}" ${f === XFER.dest ? "selected" : ""}>${f ? "loras/" + esc(f) : "loras (not sorted)"}</option>`).join("");
}
$("loraDest").addEventListener("change", (e) => { XFER.dest = e.target.value; });
$("loraLink").addEventListener("input", (e) => { $("loraLinkGo").disabled = !e.target.value.trim(); });
$("loraLink").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.stopPropagation(); ACT.linkLora(); } });
$("loraFile").addEventListener("change", (e) => { const f = [...e.target.files]; e.target.value = ""; uploadLoras(f); });
{
  const box = document.querySelector("#loraModal .lm");
  let depth = 0;
  const files = (e) => [...(e.dataTransfer?.items || [])].some((i) => i.kind === "file");
  box.addEventListener("dragenter", (e) => { if (!files(e)) return; e.preventDefault(); depth++; box.classList.add("drop"); });
  box.addEventListener("dragover", (e) => { if (files(e)) e.preventDefault(); });
  box.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; box.classList.remove("drop"); } });
  box.addEventListener("drop", (e) => {
    e.preventDefault(); e.stopPropagation(); depth = 0; box.classList.remove("drop");
    uploadLoras([...e.dataTransfer.files]);
  });
}
async function uploadLoras(files) {
  if (!files.length) return;
  const bad = files.filter((f) => !/\.safetensors$/i.test(f.name));
  if (bad.length) say(bad.length === files.length ? "Only .safetensors LoRAs can be added." : `Skipped ${bad.map((f) => f.name).join(", ")}: only .safetensors files.`);
  for (const file of files.filter((f) => !bad.includes(f))) await uploadLora(file, XFER.dest ?? LORA_FOLDER[loraTab()]);
}
async function uploadLora(file, folder) {
  // A quick look at the start, so a wrong file is turned down before it's sent.
  try {
    const head = new DataView(await file.slice(0, 9).arrayBuffer());
    const n = head.byteLength >= 9 ? Number(head.getBigUint64(0, true)) : 0;
    if (n < 2 || n + 8 > file.size || head.getUint8(8) !== 0x7b) { say(file.name + " isn't a safetensors file."); return; }
  } catch { /* read later by the server */ }
  const up = { kind: "up", at: Date.now() / 1000, key: "u" + Math.random().toString(36).slice(2), name: file.name, folder, size: file.size, sent: 0, speed: 0, state: "starting", error: "", ctl: null };
  XFER.ups.unshift(up); renderXfer();
  let overwrite = false;
  try {
    const r = await api("POST", "lora-upload", { name: file.name, folder, size: file.size });
    up.id = r.id;
    if (r.exists) {
      overwrite = await askConfirm("Replace " + file.name + "?", `There's already a LoRA called ${file.name} in ${folder ? "loras/" + folder : "loras"}. Upload this one in its place?`, "Replace");
      if (!overwrite) { cancelUpload(up); XFER.ups = XFER.ups.filter((u) => u !== up); renderXfer(); return; }
    }
    up.state = "sending"; renderXfer();
    let tries = 0, last = performance.now(), lastSent = 0;
    while (up.sent < file.size) {
      if (up.state === "cancelled") return;
      up.ctl = new AbortController();
      try {
        const res = await fetch(`/mmflow/api/lora-upload/${up.id}?offset=${up.sent}`, {
          method: "PUT", body: file.slice(up.sent, Math.min(file.size, up.sent + UP_PIECE)), signal: up.ctl.signal,
          headers: { "Content-Type": "application/octet-stream" },
        });
        if (res.status === 404) throw Object.assign(new Error(await res.text()), { final: true });
        if (!res.ok) throw Object.assign(new Error((await res.text()) || res.statusText), { final: res.status === 400 });
        up.sent = (await res.json()).offset;
        tries = 0;
      } catch (error) {
        if (up.state === "cancelled") return;
        if (error.final || ++tries > 8) throw error;
        up.state = "retrying"; renderXfer();
        await new Promise((r) => setTimeout(r, Math.min(15000, 800 * 2 ** (tries - 1))));
        // Where the server got to (a piece may have arrived even though the answer didn't).
        up.state = "sending";
        continue;
      }
      const now = performance.now();
      if (now - last > 400) {
        const sp = ((up.sent - lastSent) * 1000) / (now - last);
        up.speed = up.speed ? 0.7 * up.speed + 0.3 * sp : sp;
        last = now; lastSent = up.sent;
      }
      renderXfer();
    }
    up.state = "checking"; renderXfer();
    const done = await api("POST", `lora-upload/${up.id}/finish`, { overwrite });
    up.state = "done"; up.file = done.file;
    loraArrived(done.file);
  } catch (error) {
    if (up.state === "cancelled") return;
    up.state = "failed"; up.error = error.message;
    if (up.id) api("DELETE", "lora-upload/" + up.id).catch(() => {});
  }
  renderXfer();
}
function cancelUpload(up, why) {
  up.state = "cancelled"; up.error = why || "Cancelled.";
  up.ctl?.abort();
  if (up.id) api("DELETE", "lora-upload/" + up.id).catch(() => {});
  renderXfer();
}
async function pollLinks() {
  clearTimeout(XFER.timer);
  try {
    const before = new Set(XFER.links.filter((j) => j.state === "done").map((j) => j.id));
    XFER.links = (await api("GET", "lora-links")).jobs;
    XFER.links.filter((j) => j.state === "done" && !before.has(j.id) && j.file).forEach((j) => XFER.fresh.add(j.file));
  } catch { /* tried again below */ }
  renderXfer();
  if (XFER.links.some((j) => j.state === "queued" || j.state === "downloading")) XFER.timer = setTimeout(pollLinks, LM.open ? 1000 : 4000);
}
// A LoRA is in: list it, and show it at the top of the list until the window is opened again.
function loraArrived(file) {
  XFER.fresh.add(file);
  fetchLoras(false).then(() => say(loraStem(file) + " is ready to add."));
}
function xferRow(x) {
  const total = x.kind === "up" ? x.size : x.total, done = x.kind === "up" ? x.sent : x.done;
  const going = ["starting", "sending", "retrying", "checking", "queued", "downloading"].includes(x.state);
  const pct = total ? Math.min(100, (100 * done) / total) : 0;
  const left = x.speed > 0 && total ? timeText((total - done) / x.speed) + " left" : "";
  const where = x.folder ? "loras/" + x.folder : "loras";
  let line;
  if (x.state === "done") line = (x.note === "already there" ? "Already there · " : "Added · ") + (x.file || where) + " · click it in the list to use it";
  else if (x.state === "failed" || x.state === "cancelled") line = x.error;
  else if (x.state === "queued") line = "Waiting for the one before";
  else if (x.state === "starting") line = "Starting…";
  else if (x.state === "checking") line = "Checking the file…";
  else if (x.state === "retrying") line = `Connection dropped at ${bytesText(done)} · trying again…`;
  else line = [total ? `${bytesText(done)} of ${bytesText(total)}` : done ? bytesText(done) : "Connecting…", x.speed > 0 ? bytesText(x.speed) + "/s" : "", left].filter(Boolean).join(" · ");
  const name = x.name || x.url.replace(/^https:\/\/(www\.)?/, "");
  const btn = going ? `<button data-act="cancelXfer" data-id="${esc(x.key || x.id)}">Cancel</button>`
    : `<button data-act="clearXfer" data-id="${esc(x.key || x.id)}" aria-label="Clear">${ICON.x}</button>`;
  const bar = going ? `<div class="lm-bar ${!total || x.state === "checking" || x.state === "queued" ? "wait" : ""}"><i style="width:${pct.toFixed(1)}%"></i></div>` : "";
  return `<div class="lm-x ${x.state}" title="${esc(x.url || x.name)}"><b>${esc(name)}</b>${btn}<small>${esc(line)}</small>${bar}</div>`;
}
function renderXfer() {
  if (!LM.open) return;
  const hidden = XFER.hide || new Set();
  $("loraXfer").innerHTML = [...XFER.ups, ...XFER.links].sort((a, b) => b.at - a.at).filter((x) => !hidden.has(x.key || x.id)).map(xferRow).join("");
}
// Capture, so the window has its keys wherever the focus is, and Escape closes it and nothing else.
document.addEventListener("keydown", (e) => {
  if (!LM.open || !$("confirm").hidden) return;
  if (e.target === $("loraLink") && e.key !== "Escape") return;
  const list = loraShown();
  if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeLoras(); }
  else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    LM.sel = Math.max(0, Math.min(list.length - 1, LM.sel + (e.key === "ArrowDown" ? 1 : -1)));
    renderLoras();
  } else if (e.key === "Enter" && list[LM.sel] && !e.target.closest?.("button")) {
    e.preventDefault();
    toggleLoraFile(list[LM.sel]);
  }
}, true);

const ACT = {
  filter: (id) => { state.filter = id; render(); },
  // Switching tabs keeps each tab's own prompt and settings.
  tab: (id) => {
    if (id === state.tab) return;
    hoverStop();
    state.picked.clear(); state.lastPicked = null;
    state.drafts[draftKey()] = state.prompt;
    state.tab = id;
    state.prompt = state.drafts[draftKey()] || "";
    state.caret = state.prompt.length;
    state.mentionClosed = true;
    state.settingsOpen = false; state.library.open = false; state.loraPicker = false;
    state.filter = "all";
    render();
  },
  recCfg: () => { state.settings.cfg = modelOf(state.settings.model).cfg; render(); },
  // Send a finished image to the Video tab as an @image, ready to animate.
  useInVideo: async (id) => {
    const c = clipById(id || state.openId);
    if (!c || c.status !== "done") return;
    try {
      const ref = await refFromClip(c.id);
      attach(ref.id);
      if (isFull()) document.exitFullscreen();
      state.view = "gallery";
      ACT.tab("video");
      const sep = state.prompt && !/\s$/.test(state.prompt) ? " " : "";
      state.prompt += sep + "@" + ref.name + " ";
      state.caret = state.prompt.length;
      render();
      $("prompt").focus();
      say("@" + ref.name + " is in the Video tab's prompt. Describe what happens next.");
    } catch (error) {
      say(error.message);
    }
  },
  // A click opens the clip; with ⌘ or Ctrl (add/take out) or Shift (a range) it selects instead, and while
  // anything is selected a plain click selects too.
  pickMotion: () => {
    state.pickingMotion = !state.pickingMotion;
    render();
    if (state.pickingMotion) say("Click the video to take the motion from (Esc to stop).");
  },
  clearMotion: () => setMotion(null),
  openPeople: () => openPeople(),
  removePerson: (i) => { state.motionPeople = state.motionPeople.filter((_, k) => k !== Number(i)); PEOPLE.check = null; drawPeople(); saveUi(); },
  peopleCheck: () => checkPeople(),
  peopleClear: () => { state.motionPeople = []; PEOPLE.check = null; drawPeople(); saveUi(); },
  peopleDone: () => { $("peopleModal").hidden = true; render(); },
  menuUseMotion: () => { const c = motionSource(closeMenu()); if (c) setMotion(c.id); },
  motionTemplate: async () => {
    if (state.prompt.trim() && !confirm("Replace your prompt with the author's Motion Transfer prompt?")) return;
    try { const st = await api("GET", "swap"); state.prompt = st.motionTemplate; state.caret = state.prompt.length; render(); saveUi(); }
    catch (error) { say(error.message); }
  },
  open: (id, el, e) => {
    if (state.pickingMotion) {
      if (motionSource(id)) setMotion(id); else say("That one isn't a finished video.");
      return;
    }
    if (e && (e.metaKey || e.ctrlKey || e.shiftKey || state.picked.size)) { pickClip(id, e); return; }
    openClip(id);
  },
  back: () => {
    if (isFull()) document.exitFullscreen();
    state.view = "gallery";
    render();
  },
  appMenu: () => showAppMenu($("appMenu").hidden),
  close: async () => {
    showAppMenu(false);
    const busy = state.clips.filter(isBusy).length;
    const yes = await askConfirm("Exit MiniMax H3 Vision?", "You'll go back to ComfyUI. " + (busy
      ? (busy === 1 ? "The clip that is rendering keeps going" : "The " + busy + " clips that are rendering keep going") + " in the background, and everything is saved."
      : "Everything is saved, so you can pick up where you left off."), "Exit");
    if (yes) window.parent.postMessage("mmflow-close", location.origin);
    else $("closeBtn").focus();
  },
  // Add image's preview, as big as the window: a click or Esc closes it.
  zoomPic: (id, el) => {
    const img = el.querySelector("img");
    if (!img) return;
    $("zoomImg").src = img.src; $("zoomImg").alt = img.alt;
    const r = state.refs.find((x) => x.id === state.library.selected), c = isGen() ? clipById(state.library.selectedGen) : null;
    const w = img.naturalWidth, h = img.naturalHeight;
    $("zoomCap").textContent = (c ? c.prompt : r ? "@" + r.name : "") + (w ? `  ·  ${w}×${h}` : "");
    $("zoom").hidden = false;
    $("zoom").focus();
  },
  zoomClose: () => { $("zoom").hidden = true; $("zoomImg").removeAttribute("src"); },
  confirmYes: () => answerConfirm(true),
  confirmNo: () => answerConfirm(false),
  // The grid's selection: the box on a tile, ⌘/Ctrl-click to add or take out one, Shift-click for a range.
  pick: (id, el, e) => pickClip(id, e),
  clearPicked: () => { state.picked.clear(); state.lastPicked = null; render(); },
  deletePicked: async () => {
    const ids = pickedHere();
    if (!ids.length) return;
    const images = imageTab(), n = ids.length;
    const what = images ? (n === 1 ? "image" : "images") : n === 1 ? "video" : "videos";
    const busy = ids.filter((id) => isBusy(clipById(id))).length;
    const yes = await askConfirm(`Delete ${n === 1 ? "this " + what : n + " " + what}?`,
      `${n === 1 ? "It is" : "They are"} deleted from the output folder too. This can't be undone.`
      + (busy ? ` ${busy === 1 ? "One is" : busy + " are"} still rendering and will be stopped.` : ""), "Delete", true);
    if (!yes) return;
    ids.forEach((id) => { deleted.add(id); state.picked.delete(id); });
    state.clips = state.clips.filter((c) => !ids.includes(c.id));
    state.lastPicked = null;
    render();
    let failed = 0;
    for (const id of ids) { try { await api("DELETE", "clips/" + id); } catch { failed++; } }
    say(failed ? `Deleted ${n - failed} of ${n}; ${failed} couldn't be deleted.` : `${n === 1 ? (images ? "Image" : "Video") : n + " " + what} deleted.`);
  },
  // Not installed: open the Models page at this tab's models.
  whyMissing: () => { say(missingWhy()); openModels(state.tab); },
  openModels: () => { showAppMenu(false); openModels(); },
  openBuilder: () => { ACT.closePopups?.(); openBuilder(); },
  makeRefmod: () => openRefmodMaker(),
  menuRefmod: () => { const c = motionSource(closeMenu()); if (c) openRefmodMaker(c.id); },
  refmodSync: () => { RM.drive = { ...RM.drive, state: "syncing" }; render(); loadRefmods(true); },
  rmGo: () => makeRefmod(),
  rmClose: () => { $("refmodModal").hidden = true; },
  removeRefmod: (file) => { const s = state.videoSettings; s.refmods = (s.refmods || []).filter((m) => m.file !== file); render(); saveUi(); },
  toggleSent: () => { if (SENT.pinned) hideSent(true); else { SENT.key = ""; showSent(true); } },
  copySent: () => { const t = [...document.querySelectorAll("#sentBox .sent-text")].map((x) => x.textContent).join("\n\n"); copyText(t, () => say("Copied what the model gets.")); },
  tokenSave: async () => {
    const token = $("hfToken").value.trim();
    if (!token) { say("Paste your Hugging Face token first."); $("hfToken").focus(); return; }
    $("hfTokenGo").disabled = true; $("hfTokenGo").textContent = "Checking…";
    try {
      MODELS.data = await api("POST", "models-token", { token, remember: $("hfRemember").checked });
      const t = MODELS.data.token;
      say(`Token works${t.user ? " (" + t.user + ")" : ""}: gated models can be downloaded now.`);
      renderModels(false);
    } catch (error) {
      say(error.message);
      $("hfTokenGo").disabled = false; $("hfTokenGo").textContent = "Use token";
    }
  },
  tokenForget: async () => {
    try { MODELS.data = await api("DELETE", "models-token"); say("The token is removed."); renderModels(false); }
    catch (error) { say(error.message); }
  },
  openDrive: () => { showAppMenu(false); openDrive(); },
  closeDrive: () => closeDrive(),
  driveRefresh: () => driveFiles(true),
  driveOpen: async (name) => {
    try { driveJobs((await api("POST", "gdrive/open", { name })).jobs); say("Opening " + name + " from Google Drive…"); }
    catch (error) { say(error.message); }
  },
  driveCancel: (id) => api("POST", `gdrive/${id}/cancel`).then((r) => driveJobs(r.jobs)).catch((e) => say(e.message)),
  toggleRealism: () => { const r = realismOf(state.settings); r.on = !r.on; saveUi(); render(); },
  swapThis: () => { const c = clipById(state.openId); if (c && c.status === "done" && c.file) openSwap(c); },
  menuSwap: () => { const id = closeMenu(); const c = clipById(id); if (c && c.status === "done" && c.file) openSwap(c); },
  swapMode: (id) => { SWAP.mode = id; drawSwap(); },
  swapRes: (id) => { SWAP.res = Number(id); drawSwap(); },
  swapPick: (id) => { SWAP.picture = id; drawSwap(); },
  swapGo: () => runSwap(),
  swapWho: (id) => { SWAP.who = id; drawSwap(); },
  swapOpacity: (id) => { SWAP.opacity = Number(id); SWAP.check = null; drawSwap(); },
  swapCheck: () => checkMark(),
  swapClose: () => { $("swapModal").hidden = true; },
  menuFaces: () => { const id = closeMenu(); const c = clipById(id); if (c && c.status === "done" && c.file) openFaces(c); },
  facesGo: () => refineFaces(),
  facesClose: () => { $("facesModal").hidden = true; },
  facesPick: (id) => { FACES.face = FACES.face === id ? null : id; drawFacePics(); },
  menuDriveZip: () => { const id = closeMenu(); const c = clipById(id); if (c && c.status === "done" && c.file) driveSave({ clip: id }); },
  closeModels: () => closeModels(),
  modelsFilter: (id) => { MODELS.filter = id; renderModels(false); $("modelsBody").scrollTop = 0; },
  modelGet: (id) => modelAction(id, "download"),
  modelCancel: (id) => modelAction(id, "cancel"),
  modelAddOpen: () => {
    const open = $("mAddForm").hidden;
    $("mAddForm").hidden = !open; $("mAddOpen").hidden = open; $("mAddOpen").setAttribute("aria-expanded", String(open));
    $("mAddErr").textContent = "";
    if (open) $("mAddUrl").focus();
  },
  modelRemove: async (id) => {
    const g = MODELS.data && MODELS.data.groups.find((x) => x.id === id);
    if (!g || !(await askConfirm(`Remove ${g.name}?`, g.state === "missing" ? "It comes off the list." : `It comes off the list, and its file (${bytesText(g.bytes)}) is deleted from the models folder.`, "Remove", true))) return;
    try {
      MODELS.data = await api("POST", `models/${id}/remove`);
      say("Removed."); load(false); renderModels(false);
    } catch (error) { say(error.message); }
  },
  modelDelete: async (id) => {
    const g = MODELS.data && MODELS.data.groups.find((x) => x.id === id);
    if (!g || !(await askConfirm(`Delete ${g.name}?`, `Its files (${bytesText(g.bytes)}) are deleted from the models folder. Files another installed model needs are kept.`, "Delete", true))) return;
    modelAction(id, "delete");
  },
  star: (id) => {
    const c = clipById(id);
    c.starred = !c.starred;
    render();
    api("PATCH", "clips/" + id, { starred: c.starred }).catch((error) => say(error.message));
  },
  cancel: (id) => removeClip(id, "Render cancelled."),
  deleteClip: (id) => { id = id || state.openId; removeClip(id, kindOf(clipById(id) || {}) === "image" ? "Image deleted." : "Clip deleted."); },

  // The x on a prompt-bar image only takes it out of the bar; it stays in the library.
  resetPrompt: () => {
    state.prompt = ""; state.caret = 0; state.mentionClosed = true;
    if (isKlein()) state.kleinSlots = [null, null, null, null];
    else if (isPruna()) slots().fill(null);
    else if (editOn()) state.imageEditSlot = [null];
    else if (!isImage()) state.attached = [];
    render(); saveUi();
    $("prompt").focus();
    say(isImage() ? "Prompt cleared." : "Prompt and images cleared.");
  },
  removeRef: (id) => { state.attached = state.attached.filter((x) => x !== id); render(); saveUi(); },
  insertPicture: (i) => { insertWord(`<Picture ${Number(i) + 1}>`); $("prompt").focus(); },
  // A click on an image in the strip (Video) or a filled slot (Klein) puts its name in the prompt at the cursor.
  // A click on an image in the prompt bar (Video) or a filled slot (Klein) opens Add image on that picture.
  // From a slot, picking another picture there puts it in that slot.
  showRef: (id) => showInLibrary(state.refs.find((x) => x.id === id)),
  showSlot: (id) => {
    state.library.slot = Number(id);
    showInLibrary(slotRef(Number(id)));
    say(`Pick a picture to put in ${slotWord(Number(id))}.`);
  },
  // Klein's slots: the x empties one; an empty one opens Add image to fill it.
  clearSlot: (id) => {
    slots()[Number(id)] = null;
    render();
    say(isSlotTab() ? `${SlotWord(Number(id))} is empty now.` : "No picture to edit now.");
  },
  pickSlot: (id) => {
    state.library.slot = Number(id);
    if (!state.library.open) ACT.openLibrary(); else render();
    say(`Pick a picture for ${slotWord(Number(id))}, or drag one onto it.`);
  },
  kleinMatch: () => { state.kleinSettings.size = "match"; render(); },
  // A finished image into image 1 of the Klein tab (picture 1 of the Pruna tab, for Pruna's own), to keep editing it.
  editThis: (id) => editIn(id, tabOf(clipById(id || state.openId) || {}) === "pruna" ? "pruna" : "klein"),
  editInPruna: (id) => editIn(id, "pruna"),
  menuEditInPruna: () => { const id = closeMenu(); editIn(id, "pruna"); },
  pickMention: (id) => insertMention(state.tab === "video" && id[0] === "p" && !state.refs.some((r) => r.id === id) ? { picture: Number(id.slice(1)) } : state.refs.find((r) => r.id === id)),
  openLibrary: () => {
    state.library.open = !state.library.open;
    if (!state.library.open) state.library.slot = null;
    state.settingsOpen = false;
    if (!uploadItems().some((r) => r.id === state.library.selected)) state.library.selected = uploadItems()[0]?.id || null;
    if (!generatedItems().some((c) => c.id === state.library.selectedGen)) state.library.selectedGen = generatedItems()[0]?.id || null;
    render();
    if (state.library.open) $("libQuery")?.focus();
  },
  libSource: (id) => {
    state.library.source = id;
    renderLibNav(); renderLibList(); renderLibPreview();
  },
  libSelect: (id, el, e, focus) => {
    const hadFocus = focus || document.activeElement?.classList.contains("lib-item");
    if (isGen()) state.library.selectedGen = id; else state.library.selected = id;
    // Only move the highlight: redrawing the list would swallow a double-click.
    document.querySelectorAll("#libList .lib-item").forEach((el) => {
      const on = el.dataset.id === id;
      el.classList.toggle("on", on);
      el.setAttribute("aria-selected", on);
      el.tabIndex = on ? 0 : -1;
    });
    renderLibPreview();
    const item = document.querySelector(`.lib-item[data-id="${id}"]`);
    item?.scrollIntoView({ block: "nearest" });
    if (hadFocus) item?.focus();
  },
  libToggle: async () => {
    let id = state.library.selected;
    if (isGen()) {
      if (!state.library.selectedGen) return;
      try { id = (await refFromClip(state.library.selectedGen)).id; } catch (error) { say(error.message); return; }
    }
    const ref = state.refs.find((r) => r.id === id);
    if (!ref) return;
    if (editOn()) {
      if (state.imageEditSlot[0] === id) { state.imageEditSlot = [null]; render(); say("No picture to edit now."); return; }
      state.library.slot = null; state.library.open = false;
      putInSlot(ref, 0);
      return;
    }
    if (isSlotTab()) {
      const at = slots().indexOf(id);
      if (at >= 0) { slots()[at] = null; render(); say(`${SlotWord(at)} is empty now.`); return; }
      const target = state.library.slot;
      // Opened from an empty slot: fill it and close. Otherwise the next empty slot, and stay open.
      if (target != null) { state.library.slot = null; state.library.open = false; }
      putInSlot(ref, target != null ? target : -1);
      return;
    }
    if (state.attached.includes(id)) { state.attached = state.attached.filter((x) => x !== id); say("@" + ref.name + " taken out of the prompt bar."); }
    else {
      // Adding closes the window; the picture is in the prompt bar, ready to drag onto the text.
      if (state.tab === "video" && state.attached.length >= 9) { say("H3 takes at most 9 pictures."); return; }
      attach(id);
      say(state.tab === "video" ? `Added as Picture ${state.attached.length}. Write <Picture ${state.attached.length}> in the prompt (or type @) to use it.`
        : "@" + ref.name + " added to the prompt. Drag it onto the text, or type @" + ref.name + ", to use it.");
      saveUi(); ACT.closePopups(); return;
    }
    const hadFocus = document.activeElement?.classList.contains("lib-item");
    renderRefs(); renderLibNav(); renderLibList(); renderLibPreview(); saveUi();
    if (hadFocus) document.querySelector(".lib-item.on")?.focus();
  },
  closePopups: () => { state.settingsOpen = false; state.loraPicker = false; state.library.open = false; state.library.slot = null; render(); },
  // The Image tab's Edit switch: one picture to edit, with the edit LoRA's settings (30 steps, guidance 4).
  toggleEdit: () => {
    if (!isImage()) return;
    state.imageEdit = !state.imageEdit;
    const s = state.imageSettings, m = modelOf(s.model);
    s.steps = m.steps; s.cfg = m.cfg; s.sampler = ""; s.scheduler = "";
    render();
    say(state.imageEdit ? (state.imageEditSlot[0] ? "Edit is on: describe only the change." : "Edit is on: add the picture to edit, then describe the change.")
      : "Edit is off: new images again.");
  },
  toggleNegative: () => {
    state.negativeOpen = !state.negativeOpen;
    renderNegative();
    if (state.negativeOpen) $("negative").focus();
    saveUi();
  },
  toggleSettings: () => { state.settingsOpen = !state.settingsOpen; state.loraPicker = false; state.library.open = false; render(); },

  aspect: (id) => { state.settings.aspect = id; if (isSlotTab()) state.settings.size = id; render(); },
  sizeMatch: () => { state.settings.size = "match"; render(); },
  // Pruna: Image or Character sheet. Each keeps its own slots and its own text.
  prunaMode: (id) => {
    if (id === state.prunaMode) return;
    state.drafts[draftKey()] = state.prompt;
    state.prunaMode = id;
    state.prompt = state.drafts[draftKey()] || ""; state.caret = state.prompt.length;
    state.library.slot = null; state.settingsOpen = false;
    render(); saveUi();
  },
  kleinMode: (id) => {
    if (id === state.kleinMode) return;
    state.drafts[draftKey()] = state.prompt;
    state.kleinMode = id;
    state.prompt = state.drafts[draftKey()] || ""; state.caret = state.prompt.length;
    state.library.slot = null; state.settingsOpen = false;
    render(); saveUi();
  },
  toggleRefine: () => { const k = state.kleinSettings; if (!headOnly()) return; k.refine = !k.refine; render(); saveUi(); },
  paintMask: () => openPainter(),
  painterDone: () => finishPainter(),
  painterCancel: () => closePainter(),
  painterClear: () => { const c = $("paintCanvas"); c.getContext("2d").clearRect(0, 0, c.width, c.height); },
  painterErase: () => { painter.erase = !painter.erase; $("paintErase").classList.toggle("on", painter.erase); $("paintErase").setAttribute("aria-pressed", painter.erase); },
  swapSampler: (id) => { state.kleinSettings.swapSampler = id; render(); },
  compareToggle: () => { state.compare = !state.compare; renderClip(); },
  sheetPreset: (id) => { state.prunaSheet.preset = id; state.prunaSheet.views = SHEET_PRESETS[id].views.slice(); render(); saveUi(); },
  sheetView: (id) => {
    const v = state.prunaSheet.views;
    if (v.includes(id)) { if (v.length === 1) { say("Pick at least one view."); return; } v.splice(v.indexOf(id), 1); } else v.push(id);
    const order = Object.keys(SHEET_VIEWS);
    v.sort((a, b) => order.indexOf(a) - order.indexOf(b));
    state.prunaSheet.preset = Object.keys(SHEET_PRESETS).find((k) => SHEET_PRESETS[k].views.join() === v.join()) || "custom";
    render(); saveUi();
  },
  sheetBg: (id) => { state.prunaSheet.bg = id; render(); saveUi(); },
  toggleEnhance: () => {
    if (swapOn()) {
      const k = state.kleinSettings, body = state.kleinSwapSlots[0];
      if (!body) { say("Add the body picture first."); return; }
      if (!maskFor(body)) { k.headOnly = true; openPainter(); return; }
      k.headOnly = !k.headOnly; if (!k.headOnly) k.refine = false;
      say(k.headOnly ? "Only the head changes; the rest of the body picture stays exactly as it was." : "The whole picture is made again.");
      render(); saveUi(); return;
    }
    if (!OPT.pruna.enhance && !state.prunaSettings.enhance) { say("Enhance prompt needs Pruna's prompt upsampler (p-h3-image_prompt_upsampler_bf16.safetensors in models/text_encoders)."); return; }
    state.prunaSettings.enhance = !state.prunaSettings.enhance;
    say(state.prunaSettings.enhance ? "Enhance prompt is on: Pruna's upsampler rewrites your text into a long, detailed prompt first." : "Enhance prompt is off: your text goes to the model as it is.");
    render(); saveUi();
  },
  recSteps: () => { state.settings.steps = modelOf(state.settings.model).steps; render(); },
  nativeRes: () => setVideoRes(1.0),
  editValue: (field) => editValue(field),
  seedMode: (id) => { state.settings.seedMode = id; render(); },
  rollSeed: () => { state.settings.seed = newSeed(); render(); },
  loraPicker: () => openLoras(),
  closeLoras: () => closeLoras(),
  refreshLoras: () => fetchLoras(true),
  uploadLora: () => $("loraFile").click(),
  linkLora: async () => {
    const url = $("loraLink").value.trim();
    if (!url) return;
    $("loraLinkGo").disabled = true;
    try {
      XFER.links = (await api("POST", "lora-links", { url, folder: XFER.dest ?? LORA_FOLDER[loraTab()] })).jobs;
      $("loraLink").value = "";
      pollLinks();
    } catch (error) {
      say(error.message);
      $("loraLinkGo").disabled = false;
    }
    renderXfer();
  },
  cancelXfer: (id) => {
    const up = XFER.ups.find((u) => u.key === id);
    if (up) return cancelUpload(up);
    api("POST", `lora-links/${id}/cancel`).then((r) => { XFER.links = r.jobs; renderXfer(); }).catch((e) => say(e.message));
  },
  clearXfer: (id) => {
    XFER.ups = XFER.ups.filter((u) => u.key !== id);
    (XFER.hide ||= new Set()).add(id);
    renderXfer();
  },
  pickLora: (name) => toggleLoraFile(name),
  toggleLora: (id) => { const l = state.settings.loras.find((x) => x.id === id); l.on = !l.on; render(); },
  removeLora: (id) => { state.settings.loras = state.settings.loras.filter((x) => x.id !== id); render(); },
  generate: () => generate(),

  play: () => { const v = $("pv"); if (v) (v.paused ? v.play() : v.pause()); },
  fullscreen: () => {
    if (isFull()) document.exitFullscreen();
    else $("clipView").requestFullscreen?.().catch(() => say("Full screen isn't available here."));
  },
  // The text Pruna's upsampler wrote, as the prompt (with Enhance off), with the same pictures in picture 1, 2 …
  useUpsampled: () => {
    const c = clipById(state.openId);
    if (!c || !c.upsampled) return;
    reusePruna(c);
    if (sheetOn()) { const list = state.prunaSheetSlots.slice(); ACT.prunaMode("image"); state.prunaSlots = list.concat(Array(5).fill(null)); }
    state.prunaSettings.enhance = false;
    state.prompt = c.upsampled; state.caret = state.prompt.length;
    render(); saveUi();
    say("The upsampled prompt is in the prompt bar, with Enhance prompt off.");
  },
  copyPrompt: () => copyText(clipById(state.openId).prompt || "", "Prompt copied."),
  copySeed: () => {
    const c = clipById(state.openId);
    navigator.clipboard?.writeText(String(c.seed)).catch(() => {});
    say("Seed " + c.seed + " copied.");
  },
  reuse: (id) => {
    const c = clipById(id || state.openId);
    if (tabOf(c) === "klein") { reuseKlein(c); return; }
    if (tabOf(c) === "pruna") { reusePruna(c); return; }
    if (kindOf(c) !== state.tab) ACT.tab(kindOf(c));
    const model = tabOpt().models.some((m) => m.name === c.model) ? c.model : state.settings.model;
    Object.assign(state.settings, { model, steps: c.steps, res: c.res, aspect: c.aspect, seedMode: "random",
      loras: c.loras.map((l) => ({ id: l.name, name: l.name, strength: l.strength, on: true })) },
      kindOf(c) === "image" ? { cfg: c.cfg } : { len: c.len, refmods: (c.refmods || []).filter((m) => RM.files.includes(m.file)),
        realism: { on: !!c.realism, strength: c.realism || realismOf(state.settings).strength } },
      { sampler: c.sampler || "", scheduler: c.scheduler || "" });
    if (kindOf(c) === "video") state.motion = c.motion && motionSource(c.motion.from) ? c.motion.from : null;
    if (c.motion?.method) state.motionMethod = c.motion.method;
    if (kindOf(c) === "image") {
      state.negative = c.negative != null ? c.negative : negativeDefault(c.model);
      // An edit comes back as an edit, with its picture when it's still in Add image.
      state.imageEdit = !!c.edit && !!OPT.image.edit?.available;
      if (c.edit) state.imageEditSlot = [state.refs.some((r) => r.id === c.edit.id) ? c.edit.id : null];
    }
    state.prompt = c.prompt; state.caret = c.prompt.length; state.mentionClosed = true;
    // Only this clip's reference images: the ones it recorded, else the @names in its prompt.
    if (kindOf(c) === "video") {
      state.attached = [];
      if (Array.isArray(c.pictures)) c.pictures.forEach((p) => { if (state.refs.some((x) => x.id === p.id)) attach(p.id); });
      else if (Array.isArray(c.subjects)) c.subjects.forEach((x) => (x.images || []).forEach((p) => { if (state.refs.some((r) => r.id === p.id)) attach(p.id); }));
      else {
        (c.refs || []).forEach((r) => { if (state.refs.some((x) => x.id === r.id)) attach(r.id); });
        mentionsIn(c.prompt).forEach((n) => { const r = state.refs.find((x) => x.name === n); if (r) attach(r.id); });
      }
    }
    if (isFull()) document.exitFullscreen();
    state.view = "gallery";
    render();
    say("Prompt and settings are back in the prompt bar, with a random seed."
      + (c.edit && !state.imageEditSlot[0] ? " Its picture isn't in Add image any more: add the picture to edit." : ""));
  },
  keepSeed: () => {
    const c = clipById(state.openId);
    const s = { image: state.imageSettings, klein: state.kleinSettings, pruna: state.prunaSettings }[tabOf(c)] || state.videoSettings;
    s.seedMode = "fixed"; s.seed = c.seed;
    saveUi();
    say("New " + (tabOf(c) === "klein" ? "Klein images" : tabOf(c) === "pruna" ? "Pruna images" : kindOf(c) === "image" ? "images" : "clips") + " now use seed " + c.seed + ".");
  },
  download: (id) => {
    const c = clipById(id || state.openId);
    if (!c || c.status !== "done") { say("This is still rendering."); return; }
    const a = document.createElement("a");
    a.href = clipUrl(c);
    a.download = downloadName(c);
    document.body.appendChild(a);
    a.click();
    a.remove();
  },
  menuDownload: () => { const id = closeMenu(); ACT.download(id); },
  menuAddToEdit: () => {
    const id = closeMenu(), c = clipById(id);
    if (!c || c.status !== "done") return;
    if (EDIT.append(id)) say("Added to the end of the edit.");
    else say("It's in the edit already.");
  },
  menuExportZip: () => { const id = closeMenu(); ACT.exportZip(id); },
  exportZip: async (id) => {
    const c = clipById(id || state.openId);
    if (!c || !c.file) return;
    say("Saving " + c.file.replace(/\.\w+$/, ".zip") + "…");
    try {
      const r = await api("POST", "clips/" + encodeURIComponent(c.id) + "/export", {});
      const a = document.createElement("a");
      a.href = fileUrl(r.subfolder, r.file); a.download = r.file;
      document.body.appendChild(a); a.click(); a.remove();
      say(`Saved ${r.file}: the video and ${r.images} reference image${r.images === 1 ? "" : "s"}, in ${r.subfolder}.`);
    } catch (error) { say(error.message); }
  },
  menuDelete: () => { const id = closeMenu(); ACT.deleteClip(id); },
  menuReuse: () => { const id = closeMenu(); ACT.reuse(id); },
  menuCopyPrompt: () => { const c = clipById(closeMenu()); if (c) copyText(c.prompt || "", "Prompt copied."); },
  menuUseInVideo: () => { const id = closeMenu(); ACT.useInVideo(id); },
  menuEditThis: () => { const id = closeMenu(); ACT.editThis(id); },
};

const CHANGE = {
  addRefmod: (el) => {
    if (!el.value) return;
    const s = state.videoSettings;
    s.refmods = [...(s.refmods || []), { file: el.value, strength: 1 }].slice(0, 8);
    render(); saveUi();
  },
  renameRef: async (el) => {
    const r = state.refs.find((x) => x.id === el.dataset.id);
    const next = el.value.replace(/[^A-Za-z0-9_]/g, "");
    if (!next || state.refs.some((x) => x !== r && x.name === next)) { el.value = r.name; say("Names need letters, digits or _ and must be unique."); return; }
    try {
      await api("PATCH", "images/" + r.id, { name: next });
    } catch (error) {
      el.value = r.name; say(error.message); return;
    }
    state.prompt = state.prompt.replace(new RegExp("@" + r.name + "(?![A-Za-z0-9_])", "g"), "@" + next);
    r.name = next;
    render();
  },
  // Upload in Add image: into the slots on Klein, Pruna and Edit (the one Add image was opened from, or the
  // empty ones), into the image strip on the Video tab.
  addImages: (el) => {
    const files = [...el.files];
    el.value = "";
    if (hasSlots()) {
      const slot = state.library.slot;
      // From a slot: the picture goes there, and Add image closes, as when picking one.
      if (slot != null) { state.library.slot = null; state.library.open = false; }
      uploadToSlots(files, slot != null ? slot : -1);
      return;
    }
    uploadImages(files);
  },
  importFiles: (el) => { importFiles([...el.files]); el.value = ""; },
  libSort: (el) => { state.library.sort = el.value; renderLibList(); },
  // A new model starts at its recommended steps (and, for images, its recommended guidance).
  model: (el) => {
    const m = modelOf(el.value);
    state.settings.model = el.value; state.settings.steps = m.steps;
    state.settings.sampler = ""; state.settings.scheduler = "";   // back to what the new model is tuned for
    if (isImage() || isKlein()) state.settings.cfg = m.cfg;
    if (isPruna()) { state.settings.res = prunaFit(m, state.settings.res); state.settings.sheetRes = prunaFit(m, state.settings.sheetRes); }
    render();
  },
  res: (el) => { state.settings[sheetOn() ? "sheetRes" : swapOn() ? "swapRes" : "res"] = Number(el.value); render(); },
  swapLora: (el) => { state.kleinSettings.swapLora = el.value; render(); },
  swapModel: (el) => { SWAP.model = el.value; drawSwap(); },
  motionMethod: (el) => { state.motionMethod = el.value; render(); saveUi(); },
  peoplePicture: (el) => { state.motionPeople[Number(el.dataset.index)].picture = Number(el.value); saveUi(); },
  sheetRole: (el) => { state.prunaSheet.roles[Number(el.dataset.id)] = el.value; render(); saveUi(); },
  sampler: (el) => { state.settings.sampler = el.value; render(); },
  scheduler: (el) => { state.settings.scheduler = el.value; render(); },
  seed: (el) => { const v = el.value.replace(/\D/g, "").slice(0, 10); state.settings.seed = v ? Number(v) : 0; render(); },
};

const INPUT = {
  // Face swap's sliders: LanPaint steps, the mask's grow and soft edge, Refine face's strength.
  swapNum: (el) => {
    const k = state.kleinSettings, key = el.dataset.id;
    k[key] = key === "refineStrength" ? Number(el.value) / 100 : Number(el.value);
    const out = document.querySelector(`[data-val="${key}"]`);
    if (out) out.textContent = key === "refineStrength" ? el.value + "%" : key === "lanSteps" ? el.value : el.value + " px";
  },
  steps: (el) => {
    const s = state.settings, rec = modelOf(s.model).steps;
    s.steps = Number(el.value);
    $("stepsVal").textContent = stepsText(s.steps, rec);
    $("stepsVal").classList.toggle("rec", s.steps === rec);
    $("recBtn").style.visibility = s.steps === rec ? "hidden" : "visible";
    $("pill").textContent = summary();
    saveUi();
  },
  cfg: (el) => {
    const s = state.settings, rec = modelOf(s.model).cfg;
    s.cfg = Number(el.value);
    $("cfgVal").textContent = cfgText(s.cfg, rec);
    $("cfgVal").classList.toggle("rec", s.cfg === rec);
    $("cfgBtn").style.visibility = s.cfg === rec ? "hidden" : "visible";
    $("pill").textContent = summary();
    saveUi();
  },
  refmodStrength: (el) => {
    const m = (state.videoSettings.refmods || []).find((x) => x.file === el.dataset.id);
    if (!m) return;
    m.strength = Number(el.value);
    const v = document.querySelector(`[data-refmod="${CSS.escape(m.file)}"]`);
    if (v) v.textContent = m.strength.toFixed(2);
    saveUi();
  },
  resSlide: (el) => {
    const s = state.settings;
    s.res = tabOpt().megapixels[Number(el.value)];
    $("resVal").textContent = resText(s);
    $("resVal").classList.toggle("rec", s.res === 1);
    el.setAttribute("aria-valuetext", resText(s));
    document.querySelector('[data-act="nativeRes"]').style.visibility = s.res === 1 ? "hidden" : "visible";
    $("pill").textContent = summary(); saveUi();
  },
  len: (el) => { state.settings.len = Number(el.value); $("lenVal").textContent = el.value + " s"; $("pill").textContent = summary(); saveUi(); },
  realismStrength: (el) => {
    realismOf(state.settings).strength = Number(el.value);
    $("realismVal").textContent = Number(el.value).toFixed(2);
    saveUi();
  },
  loraStrength: (el) => {
    const l = state.settings.loras.find((x) => x.id === el.dataset.id);
    l.strength = Number(el.value);
    document.querySelector(`[data-strength="${CSS.escape(l.id)}"]`).textContent = l.strength.toFixed(2);
    $("pill").textContent = summary();
    saveUi();
  },
  libQuery: (el) => { state.library.query = el.value; renderLibList(); },
  perRow: (el) => {
    state.perRow = Number(el.value);
    $("grid").style.setProperty("--per-row", state.perRow);
    $("perRowVal").textContent = state.perRow;
    saveUi();
  },
};

function fromImage(image) {
  return { ...image, img: image.url || viewUrl("images", image.file), tone: toneOf(image.id) };
}

// Use this prompt and settings, for a Klein image: its prompt, settings and the pictures it was made from, each
// back in its own slot (when they are still in the Add image library).
function reuseKlein(c) {
  if (!isKlein()) ACT.tab("klein");
  const s = state.kleinSettings;
  const model = OPT.klein.models.some((m) => m.name === c.model) ? c.model : s.model;
  Object.assign(s, { model, steps: c.steps, cfg: c.cfg, res: c.res, size: c.size || "match", seedMode: "random",
    loras: c.loras.map((l) => ({ id: l.name, name: l.name, strength: l.strength, on: true })) });
  if (c.size && c.size !== "match") s.aspect = c.size;
  if (swapOn() !== !!c.swap) ACT.kleinMode(c.swap ? "swap" : "edit");
  const slots = (c.slots || []).concat([null, null, null, null, null]).slice(0, c.swap ? 5 : 4);
  const kept = slots.map((x) => (x && state.refs.some((r) => r.id === x.id) ? x.id : null));
  if (c.swap) {
    state.kleinSwapSlots = kept;
    Object.assign(s, { swapRes: c.res, swapLora: c.bfs || s.swapLora, headOnly: !!c.mask, swapSampler: c.sampler || "standard",
      lanSteps: c.lanSteps || 20, refine: !!c.refine, refineStrength: c.refineStrength || 0.5 });
    if (c.mask && kept[0]) state.kleinSwapMask = { image: kept[0], mask: c.mask.file.split("/").pop(), url: viewUrl("masks", c.mask.file.split("/").pop()) };
  } else state.kleinSlots = kept;
  const lost = slots.filter((x, i) => x && !kept[i]).length;
  state.prompt = c.swap && c.prompt === "Face swap" ? "" : c.prompt; state.caret = state.prompt.length; state.mentionClosed = true;
  if (isFull()) document.exitFullscreen();
  state.view = "gallery";
  render(); saveUi();
  say("Prompt, settings and pictures are back, with a random seed." + (lost ? ` ${lost === 1 ? "One picture isn't" : lost + " pictures aren't"} in the Add image library any more.` : ""));
}

// A finished image into slot 1 of the Klein tab (image 1) or of the Pruna tab's Image (picture 1).
async function editIn(id, tab) {
  const c = clipById(id || state.openId);
  if (!c || c.status !== "done") return;
  try {
    const ref = await refFromClip(c.id);
    if (isFull()) document.exitFullscreen();
    state.view = "gallery";
    if (state.tab !== tab) ACT.tab(tab);
    if (tab === "pruna") { if (sheetOn()) ACT.prunaMode("image"); state.prunaSlots[0] = ref.id; }
    else slots()[0] = ref.id;
    render();
    $("prompt").focus();
    say(`It's ${tab === "pruna" ? "picture 1" : "image 1"} now. Describe the next edit.`);
  } catch (error) {
    say(error.message);
  }
}

// A Pruna image's prompt, settings and pictures back in the Pruna tab (its sheet's views, for a sheet).
function reusePruna(c) {
  if (!isPruna()) ACT.tab("pruna");
  const sheet = c.mode === "sheet";
  if (sheetOn() !== sheet) ACT.prunaMode(sheet ? "sheet" : "image");
  const s = state.prunaSettings;
  Object.assign(s, { model: OPT.pruna.models.some((m) => m.name === c.model) ? c.model : s.model, seedMode: "random", enhance: !!c.enhance && !!OPT.pruna.enhance,
    loras: c.loras.map((l) => ({ id: l.name, name: l.name, strength: l.strength, on: true })) });
  s[sheet ? "sheetRes" : "res"] = c.res;
  if (!sheet) { s.size = c.size && c.size !== "match" ? c.size : "match"; if (c.size && c.size !== "match") s.aspect = c.size; }
  const count = sheet ? 4 : 9, saved = (c.slots || []).concat(Array(count).fill(null)).slice(0, count);
  const list = saved.map((x) => (x && state.refs.some((r) => r.id === x.id) ? x.id : null));
  if (sheet) {
    state.prunaSheetSlots = list;
    if (c.sheet) Object.assign(state.prunaSheet, { views: c.sheet.views.slice(), preset: c.sheet.preset || "custom", bg: c.sheet.bg || "white", roles: (c.sheet.roles || state.prunaSheet.roles).slice() });
  } else state.prunaSlots = list;
  const lost = saved.filter((x, i) => x && !list[i]).length;
  state.prompt = sheet && c.prompt === "Character sheet" ? "" : c.prompt; state.caret = state.prompt.length; state.mentionClosed = true;
  if (isFull()) document.exitFullscreen();
  state.view = "gallery";
  render(); saveUi();
  say("Prompt, settings and pictures are back, with a random seed." + (lost ? ` ${lost === 1 ? "One picture isn't" : lost + " pictures aren't"} in the Add image library any more.` : ""));
}

// Saved files are named vision_<start of the prompt>; older ones get that name when downloaded.
const slugOf = (text) => {
  let out = "";
  for (const w of (text || "").toLowerCase().replace(/'/g, "").match(/[a-z0-9]+/g) || []) {
    if (out && out.length + 1 + w.length > 40) break;
    out = out ? out + "_" + w : w.slice(0, 40);
  }
  return out || "untitled";
};
function downloadName(c) {
  if (c.file && c.file.startsWith("vision_")) return c.file.replace(/_(\d{5})_(\.\w+)$/, "_$1$2");
  const ext = (c.file && c.file.match(/\.\w+$/)?.[0]) || (kindOf(c) === "image" ? ".png" : ".mp4");
  return "vision_" + slugOf(c.prompt) + "_" + c.seed + ext;
}

// Videos and images made earlier: the server reads the prompt and settings saved inside them.
const isZip = (f) => /\.zip$/i.test(f.name);

// What a zip import brought in: its videos join the grid, its reference pictures the Add image library.
function importedZip(r) {
  for (const c of [...r.clips].reverse()) if (!clipById(c.id)) state.clips.unshift(c);
  for (const i of r.images || []) if (!state.refs.some((x) => x.id === i.id)) state.refs.push(fromImage(i));
  return r.clips;
}
function zipSummary(r) {
  const n = r.clips.length, pics = (r.images || []).length;
  return `${n} video${n === 1 ? "" : "s"} added` + (r.same ? `, ${r.same} already here` : "")
    + (pics || r.sameImages ? `; ${pics} reference image${pics === 1 ? "" : "s"} added` + (r.sameImages ? `, ${r.sameImages} already here` : "") : "");
}

async function importFiles(files) {
  files = files.filter((f) => f.type.startsWith("video/") || f.type.startsWith("image/") || isZip(f));
  if (!files.length) return;
  const done = [], failed = [], zips = [];
  for (const [i, f] of files.entries()) {
    say(files.length === 1 ? "Reading " + f.name + "…" : `Importing ${i + 1} of ${files.length}…`);
    const form = new FormData();
    form.append("file", f, f.name);
    try {
      if (isZip(f)) {
        const r = await api("POST", "import-zip", form);
        zips.push(r);
        const here = r.clips.length ? r.clips : [];
        importedZip(r);
        done.push(...here);
        continue;
      }
      const { clip } = await api("POST", "import", form);
      if (!clipById(clip.id)) state.clips.unshift(clip);
      done.push(clip);
    } catch (error) {
      failed.push(error.message);
    }
  }
  if (zips.length && !failed.length) {
    const sum = { clips: zips.flatMap((r) => r.clips), images: zips.flatMap((r) => r.images || []),
      same: zips.reduce((a, r) => a + (r.same || 0), 0), sameImages: zips.reduce((a, r) => a + (r.sameImages || 0), 0) };
    if (state.tab === "edit") EDIT.clipsChanged();
    else if (done.length === 1) { if (state.tab !== "video") ACT.tab("video"); openClip(done[0].id); }
    else { if (state.tab !== "video") ACT.tab("video"); render(); }
    say((files.length === 1 ? "Imported " + files[0].name + ": " : "Imported: ") + zipSummary(sum) + "."
      + (zips.some((r) => r.edit) ? " Open a project's timeline from Edit › Project › Import project." : ""));
    return;
  }
  if (done.length) {
    // In the Edit tab, imported videos just join its media panel.
    if (state.tab === "edit" && done.every((c) => kindOf(c) === "video")) EDIT.clipsChanged();
    else {
      if (tabOf(done[done.length - 1]) !== state.tab) ACT.tab(tabOf(done[done.length - 1]));
      if (done.length === 1) openClip(done[0].id); else render();
    }
  }
  if (failed.length) say(failed.length === 1 ? failed[0] : `${failed.length} files couldn't be imported. ${failed[0]}`);
  else if (done.length === 1) say(done[0].found === "graph" ? "Imported. The settings were worked out from the file's ComfyUI workflow." : "Imported with its prompt and settings.");
  else if (done.length) say("Imported " + done.length + " files.");
}

async function uploadImages(files) {
  files = files.filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("file", f, f.name));
  say(files.length === 1 ? "Uploading…" : "Uploading " + files.length + " images…");
  try {
    const { images } = await api("POST", "images", form);
    const added = images.map(fromImage);
    state.refs.push(...added);
    // Only the Video tab has an image strip: elsewhere the pictures just join Add image.
    const video = state.tab === "video";
    if (video) added.forEach((a) => attach(a.id));
    state.library.selected = added[0]?.id || state.library.selected;
    render();
    document.querySelector(".lib-item.on")?.scrollIntoView({ block: "nearest" });
    say(video ? "Added @" + added.map((a) => a.name).join(", @") + " to the prompt bar. Drag it onto the text, or type @, to use it."
      : added.length === 1 ? "@" + added[0].name + " is in Add image." : added.length + " pictures are in Add image.");
  } catch (error) {
    say("Upload failed: " + error.message);
  }
}

document.addEventListener("click", (e) => {
  // A click anywhere outside the gear menu closes it.
  if (!$("appMenu").hidden && !e.target.closest("#appMenuWrap")) showAppMenu(false);
  const el = e.target.closest("[data-act]");
  if (el && (el.classList.contains("off") || el.disabled)) return;
  if (el && ACT[el.dataset.act]) ACT[el.dataset.act](el.dataset.id, el, e);
});
// Picking from the @ list keeps the cursor in the prompt.
document.addEventListener("mousedown", (e) => { if (e.target.closest(".mention")) e.preventDefault(); });
document.addEventListener("change", (e) => {
  const el = e.target.closest("[data-change]");
  if (el && CHANGE[el.dataset.change]) CHANGE[el.dataset.change](el);
});
document.addEventListener("input", (e) => {
  const el = e.target.closest("[data-input]");
  if (el && INPUT[el.dataset.input]) INPUT[el.dataset.input](el);
});
// Esc closes the big picture first, and nothing else.
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("zoom").hidden) { e.preventDefault(); e.stopImmediatePropagation(); ACT.zoomClose(); }
  else if (e.key === "Escape" && !$("models").hidden && $("confirm").hidden) { e.preventDefault(); e.stopImmediatePropagation(); closeModels(); }
  else if (e.key === "Escape" && !$("painter").hidden) { e.preventDefault(); e.stopImmediatePropagation(); closePainter(); }
}, true);
document.addEventListener("keydown", (e) => {
  if (!$("appMenu").hidden && e.key === "Escape") { e.preventDefault(); showAppMenu(false); $("closeBtn").focus(); return; }
  if (!$("confirm").hidden) {
    if (e.key === "Escape") { e.preventDefault(); ACT.confirmNo(); }
    if (e.key === "Tab") {
      e.preventDefault();
      (document.activeElement === $("confirmYes") ? $("confirmNo") : $("confirmYes")).focus();
    }
    return;
  }
  if (e.key === "Escape" && e.defaultPrevented) return;
  if (e.key === "Escape" && e.target.id !== "prompt" && (state.settingsOpen || state.library.open)) ACT.closePopups();
  // Esc on an open video or image goes back to the grid, like Back to gallery (not while typing).
  else if (e.key === "Escape" && state.view === "clip" && !isFull()
    && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) && !e.target.isContentEditable) { e.preventDefault(); ACT.back(); }
  // Esc in the grid lets go of the selection.
  else if (e.key === "Escape" && state.view === "gallery" && state.picked.size && !/^(INPUT|TEXTAREA)$/.test(e.target.tagName)) ACT.clearPicked();
});
document.addEventListener("keydown", (e) => {
  if (!state.library.open) return;
  const item = e.target.closest?.(".lib-item");
  const inSearch = e.target.id === "libQuery";
  if (!item && !inSearch) return;
  // Up and down move through the list, from the search field too.
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (inSearch && !libraryIds().includes(libSelected())) libMove(e.key === "ArrowDown" ? 1 : -1);
    else if (inSearch) ACT.libSelect(libSelected(), null, null, true);
    else libMove(e.key === "ArrowDown" ? 1 : -1);
    return;
  }
  if (e.key === "Home" || e.key === "End") {
    if (!item) return;
    e.preventDefault();
    const ids = libraryIds();
    if (ids.length) ACT.libSelect(e.key === "Home" ? ids[0] : ids[ids.length - 1], null, null, true);
    return;
  }
  // Enter picks an image, and Enter again adds it (or takes it out). From the search field, Enter adds the selected one.
  if (e.key === "Enter" && inSearch) { e.preventDefault(); if (libSelected()) ACT.libToggle(); return; }
  if (!item || (e.key !== "Enter" && e.key !== " ")) return;
  e.preventDefault();
  if (libSelected() === item.dataset.id && e.key === "Enter") ACT.libToggle();
  else ACT.libSelect(item.dataset.id, null, null, true);
});
document.addEventListener("fullscreenchange", () => { if (state.view === "clip") { $("clipView").classList.toggle("fs", isFull()); syncFullscreenButton(); } });
// On the clip page: double-click the picture or press F for full screen, Space to play or pause.
document.addEventListener("dblclick", (e) => { if (e.target.closest(".player") && !e.target.closest("button") && ($("pv") || $("pimg"))) ACT.fullscreen(); });
document.addEventListener("keydown", (e) => {
  if (state.view !== "clip" || e.target.closest("input, textarea, select") || !($("pv") || $("pimg"))) return;
  if (e.key === "f" || e.key === "F") ACT.fullscreen();
  // Left and right go to the previous and next video or image in the gallery.
  if ((e.key === "ArrowLeft" || e.key === "ArrowRight") && !e.altKey && !e.metaKey && !e.ctrlKey) {
    const list = filteredClips(), at = list.findIndex((c) => c.id === state.openId);
    const next = list[at + (e.key === "ArrowRight" ? 1 : -1)];
    e.preventDefault();
    if (next) openClip(next.id);
    return;
  }
  if (!$("pv")) return;
  if (e.key === " " && !e.target.closest("button")) { e.preventDefault(); ACT.play(); }
});
// Double-click an image in the library to add it straight away.
document.addEventListener("dblclick", (e) => {
  const item = e.target.closest(".lib-item");
  if (!item) return;
  const refId = isGen() ? refForClip(item.dataset.id)?.id : item.dataset.id;
  if (refId && state.attached.includes(refId)) return;
  if (isGen()) state.library.selectedGen = item.dataset.id; else state.library.selected = item.dataset.id;
  ACT.libToggle();
});
window.addEventListener("resize", () => { if (state.settingsOpen) renderSettings(); if (state.library.open) renderLibrary(); });

// Image files can be dropped anywhere on the page, or pasted into the prompt.
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
document.addEventListener("dragover", (e) => {
  if (state.view !== "gallery" || !hasFiles(e)) return;
  e.preventDefault();
  if (!isImage()) $("addRef")?.classList.add("drag");
  if (hasSlots()) markReplace(e.target.closest?.("[data-slot]") || null);
});
document.addEventListener("dragleave", (e) => { if (!e.relatedTarget) $("addRef")?.classList.remove("drag"); });
document.addEventListener("drop", (e) => {
  if (state.view !== "gallery" || !hasFiles(e)) return;
  e.preventDefault();
  $("addRef")?.classList.remove("drag");
  const files = [...e.dataTransfer.files];
  if (hasSlots()) {
    // Pictures go in the slots (the one dropped on, or the empty ones); videos and zips are still imports.
    const slot = e.target.closest?.("[data-slot]");
    markReplace(null);
    const rest = files.filter((f) => f.type.startsWith("video/") || isZip(f));
    if (rest.length) importFiles(rest);
    uploadToSlots(files, slot ? Number(slot.dataset.slot) : -1);
    return;
  }
  // Videos are imports. Pictures are imports in the Image tab, and @images in the Video tab.
  const imports = files.filter((f) => f.type.startsWith("video/") || isZip(f) || (isImage() && f.type.startsWith("image/")));
  if (imports.length) importFiles(imports);
  if (!isImage() && state.tab !== "edit") uploadImages(files.filter((f) => f.type.startsWith("image/")));
});

/* ------------------------------------------------------------------
   Dragging your images into the prompt. The library list, its big preview and the
   prompt-bar images can all be dragged. Dropped on the text, the @name goes where you
   let go; dropped elsewhere on the prompt box, it goes at the end.
   ------------------------------------------------------------------ */
const REF_MIME = "application/x-minimax-ref";
const GEN_MIME = "application/x-minimax-generated";
const SLOT_MIME = "application/x-minimax-slot";   // a Klein slot's picture, dragged to another slot
const isRefDrag = (e) => [...(e.dataTransfer?.types || [])].some((t) => t === REF_MIME || t === GEN_MIME || t === SLOT_MIME);
const composer = document.querySelector(".composer");
const ta = $("prompt");

let dragFromBar = false;   // a drag that started in the prompt bar (its image strip or Klein's slots)
document.addEventListener("dragstart", (e) => {
  dragFromBar = !!e.target.closest?.("#refs");
  const slot = e.target.closest?.("[data-drag-slot]");
  if (slot) {
    e.dataTransfer.clearData();
    e.dataTransfer.setData(SLOT_MIME, slot.dataset.dragSlot);
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setDragImage(slot, 40, 40);
    slot.closest(".slot")?.classList.add("dragging");
    document.body.classList.add("dragging-ref");
    return;
  }
  const gen = e.target.closest?.("[data-drag-gen]");
  if (gen) {
    const c = clipById(gen.dataset.dragGen);
    if (!c) return;
    const ref = refForClip(c.id);
    e.dataTransfer.clearData();
    e.dataTransfer.setData(GEN_MIME, c.id);
    e.dataTransfer.setData("text/plain", ref ? "@" + ref.name + " " : c.prompt);
    e.dataTransfer.effectAllowed = "copy";
    const thumb = gen.querySelector(".lib-thumb") || gen.querySelector(".pic img");
    if (thumb) e.dataTransfer.setDragImage(thumb, 24, 24);
    document.body.classList.add("dragging-ref");
    return;
  }
  const src = e.target.closest?.("[data-drag-ref]");
  if (!src) return;
  const r = state.refs.find((x) => x.id === src.dataset.dragRef);
  if (!r) return;
  e.dataTransfer.clearData();
  e.dataTransfer.setData(REF_MIME, r.id);
  e.dataTransfer.setData("text/plain", "@" + r.name + " ");
  e.dataTransfer.effectAllowed = "copy";
  e.dataTransfer.setDragImage(src.querySelector(".lib-thumb") || src, 24, 24);
  document.body.classList.add("dragging-ref");
});
document.addEventListener("dragend", () => {
  document.body.classList.remove("dragging-ref");
  document.querySelectorAll(".slot.dragging").forEach((x) => x.classList.remove("dragging"));
  composer.classList.remove("drop-on");
  markReplace(null);
});
// An image dropped on an image in the strip (or on a missing one) replaces it, in the strip and in the prompt.
const replaceTarget = (e) => e.target.closest?.(hasSlots() ? "[data-slot]" : "[data-replace-ref], [data-replace-name]");
function markReplace(el) {
  document.querySelectorAll(".replace-on").forEach((x) => { if (x !== el) x.classList.remove("replace-on"); });
  el?.classList.add("replace-on");
}
composer.addEventListener("dragover", (e) => {
  if (!isRefDrag(e)) return;
  e.preventDefault();
  // A slot's picture moves (it is "move" only); everything else is copied in.
  e.dataTransfer.dropEffect = [...e.dataTransfer.types].includes(SLOT_MIME) ? "move" : "copy";
  composer.classList.add("drop-on");
  markReplace(replaceTarget(e));
});
composer.addEventListener("dragleave", (e) => { if (!composer.contains(e.relatedTarget)) { composer.classList.remove("drop-on"); markReplace(null); } });
composer.addEventListener("drop", async (e) => {
  if (!isRefDrag(e)) return;
  e.preventDefault();
  composer.classList.remove("drop-on");
  if (hasSlots()) { dropOnSlots(e); return; }
  const genId = e.dataTransfer.getData(GEN_MIME);
  const onText = e.target === ta;
  const target = replaceTarget(e);
  const oldId = target?.dataset.replaceRef, oldName = target?.dataset.replaceName;
  markReplace(null);
  let r;
  if (genId) { try { r = await refFromClip(genId); } catch (error) { say(error.message); return; } }
  else r = state.refs.find((x) => x.id === e.dataTransfer.getData(REF_MIME));
  if (!r) return;
  // A picture from the prompt bar: onto the text it writes its <Picture N>; onto another picture it moves there.
  // One brought in from Add image replaces the picture it's dropped on, or else joins the end.
  if (dragFromBar) {
    const from = state.attached.indexOf(r.id);
    if (onText) { if (from >= 0) insertWord(`<Picture ${from + 1}>`); return; }
    // Onto another picture: it takes that place (the ones between move along). Tags in the prompt stay as written.
    const target = e.target.closest?.("[data-replace-ref]");
    const to = target ? state.attached.indexOf(target.dataset.replaceRef) : -1;
    if (from >= 0 && to >= 0 && from !== to) {
      state.attached.splice(from, 1); state.attached.splice(to, 0, r.id);
      render(); saveUi();
      say(`Moved to Picture ${to + 1}.` + (/<Picture \d+>/.test(state.prompt) ? " Check the prompt's <Picture N> tags still point at the right pictures." : ""));
    }
    return;
  }
  if (target) {
    replaceRef(oldId, oldName, r);
    // Replaced from Add image: that's done, so it closes.
    if (state.library.open && !dragFromBar) ACT.closePopups();
    return;
  }
  const had = state.attached.includes(r.id);
  attach(r.id);
  renderRefs();
  if (state.library.open) { renderLibNav(); renderLibList(); renderLibPreview(); }
  saveUi();
  say(had ? `That picture is Picture ${state.attached.indexOf(r.id) + 1} already.` : `Added as Picture ${state.attached.indexOf(r.id) + 1}. Write <Picture ${state.attached.indexOf(r.id) + 1}> in the prompt (or type @) to use it.`);
});

// Klein: a picture dropped on a slot goes in that slot (a slot's picture swaps places with what's there); dropped
// anywhere else on the prompt bar, it goes in the first empty slot. The prompt stays as it is.
async function dropOnSlots(e) {
  const target = e.target.closest?.("[data-slot]");
  const to = target ? Number(target.dataset.slot) : -1;
  markReplace(null);
  const from = e.dataTransfer.getData(SLOT_MIME);
  if (from !== "") {
    const a = Number(from);
    // Dropped on the prompt text: its name goes where the cursor was (Klein: the pictures have names).
    if (to < 0 && e.target === ta) { if (isSlotTab() && !sheetOn() && !swapOn()) insertWord(slotWord(a)); return; }
    if (to < 0 || a === to) return;
    const list = slots();
    [list[a], list[to]] = [list[to], list[a]];
    render();
    say(list[a] ? `Swapped ${slotWord(a)} and ${slotWord(to)}.` : `Moved to ${slotWord(to)}.`);
    return;
  }
  const genId = e.dataTransfer.getData(GEN_MIME);
  let r;
  if (genId) { try { r = await refFromClip(genId); } catch (error) { say(error.message); return; } }
  else r = state.refs.find((x) => x.id === e.dataTransfer.getData(REF_MIME));
  if (!r) return;
  const replaced = (to >= 0 || editOn()) && slots()[Math.max(0, to)] && slots()[Math.max(0, to)] !== r.id;
  const at = putInSlot(r, to);
  if (replaced && at >= 0) say(isSlotTab() ? `@${r.name} is ${sheetOn() || swapOn() ? "the " : ""}${slotWord(at)} now (replaced).` : `Editing @${r.name} now (replaced).`);
  // Replaced from Add image: that's done, so it closes.
  if (replaced && state.library.open) { ACT.closePopups(); return; }
  if (state.library.open) { renderLibNav(); renderLibList(); renderLibPreview(); }
}

// Files from your computer, for Klein: into the slot they were dropped on, or the empty slots in order.
async function uploadToSlots(files, slot = -1) {
  files = files.filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  const form = new FormData();
  files.forEach((f) => form.append("file", f, f.name));
  say(files.length === 1 ? "Uploading…" : "Uploading " + files.length + " images…");
  try {
    const { images } = await api("POST", "images", form);
    const added = images.map(fromImage);
    state.refs.push(...added);
    const placed = [];
    added.forEach((r, n) => {
      const list = slots();
      const at = n === 0 && slot >= 0 ? slot : editOn() ? (n === 0 ? 0 : -1) : list.indexOf(null);
      if (at >= 0) { list[at] = r.id; placed.push(isSlotTab() ? `@${r.name} is ${sheetOn() || swapOn() ? "the " : ""}${slotWord(at)}` : `Editing @${r.name}`); }
    });
    render();
    say(placed.length ? placed.join(", ") + "." + (placed.length < added.length ? " The rest are in Add image (all the slots are full)." : "")
      : "All the slots are full: the pictures are in Add image.");
  } catch (error) {
    say("Upload failed: " + error.message);
  }
}

// Put image r where another was: in the strip at its place, and as @r.name wherever the prompt named the old one.
function replaceRef(oldId, oldName, r) {
  const old = oldId ? state.refs.find((x) => x.id === oldId) : null;
  const name = old ? old.name : oldName;
  if (old && old.id === r.id) return;
  if (old) {
    const at = state.attached.indexOf(old.id);
    state.attached = state.attached.filter((x) => x !== r.id);
    const i = state.attached.indexOf(old.id);
    if (i >= 0) state.attached.splice(i, 1, r.id); else state.attached.splice(Math.max(0, at), 0, r.id);
  } else attach(r.id);
  const re = new RegExp("@" + name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![A-Za-z0-9_])", "g");
  const count = (state.prompt.match(re) || []).length;
  state.prompt = state.prompt.replace(re, "@" + r.name);
  state.caret = Math.min(state.caret, state.prompt.length);
  state.mentionClosed = true;
  renderPrompt(); renderRefs();
  if (state.library.open) { renderLibNav(); renderLibList(); renderLibPreview(); }
  saveUi();
  say(`@${name} is now @${r.name}` + (count ? (count > 1 ? ` (${count} places in the prompt).` : " in the prompt.") : "."));
}

// A drop inside a word moves to the nearer end of that word, so words are never split.
function wordEdge(text, pos) {
  const word = /[^\s]/;
  if (!(word.test(text[pos - 1] || "") && word.test(text[pos] || ""))) return pos;
  let a = pos, b = pos;
  while (a > 0 && word.test(text[a - 1])) a--;
  while (b < text.length && word.test(text[b])) b++;
  return pos - a <= b - pos ? a : b;
}

// Which character of the prompt is under a point. The coloured copy under the text box
// has the same layout, so ask the browser for the spot in that copy.
function textOffsetAt(x, y) {
  ta.style.pointerEvents = "none";
  let node = null, offset = 0;
  if (document.caretPositionFromPoint) {
    const p = document.caretPositionFromPoint(x, y);
    if (p) { node = p.offsetNode; offset = p.offset; }
  } else if (document.caretRangeFromPoint) {
    const r = document.caretRangeFromPoint(x, y);
    if (r) { node = r.startContainer; offset = r.startOffset; }
  }
  ta.style.pointerEvents = "";
  const mirror = $("mirror");
  if (!state.prompt || !node || !mirror.contains(node)) return state.prompt.length;
  const walk = document.createTreeWalker(mirror, NodeFilter.SHOW_TEXT);
  let count = 0;
  while (walk.nextNode()) {
    if (walk.currentNode === node) return Math.min(count + offset, state.prompt.length);
    count += walk.currentNode.length;
  }
  return state.prompt.length;
}

/* The prompt box */
// Clicking into the prompt closes an open pop-up (image library or settings).
ta.addEventListener("pointerdown", () => { if (state.library.open || state.settingsOpen) ACT.closePopups(); });
ta.addEventListener("input", () => {
  state.prompt = ta.value; state.caret = ta.selectionStart;
  state.mentionClosed = false; state.mentionIndex = 0;
  renderPrompt();
  renderRefs();
  saveUi();
});
ta.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files || [])];
  if (hasSlots() && files.some((f) => f.type.startsWith("image/"))) { e.preventDefault(); uploadToSlots(files); return; }
  if (!isImage() && files.some((f) => f.type.startsWith("image/"))) { e.preventDefault(); uploadImages(files); }
});
const syncCaret = () => { if (ta.selectionStart !== state.caret) { state.caret = ta.selectionStart; renderMention(); } };
ta.addEventListener("click", syncCaret);
ta.addEventListener("keyup", (e) => { if (!["ArrowDown", "ArrowUp", "Enter", "Tab", "Escape"].includes(e.key)) syncCaret(); });
ta.addEventListener("keydown", (e) => {
  const list = mentionMatches();
  if (list.length) {
    const n = list.length, i = Math.min(state.mentionIndex, n - 1);
    if (e.key === "ArrowDown") { e.preventDefault(); state.mentionIndex = (i + 1) % n; renderMention(); return; }
    if (e.key === "ArrowUp") { e.preventDefault(); state.mentionIndex = (i - 1 + n) % n; renderMention(); return; }
    if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); insertMention(list[i]); return; }
  }
  if (e.key === "Escape") {
    // preventDefault marks the key as used, so it doesn't also leave the clip view.
    if (mentionQuery() !== null) { e.preventDefault(); state.mentionClosed = true; renderMention(); }
    else if (state.settingsOpen || state.library.open) { e.preventDefault(); ACT.closePopups(); }
  }
});

/* ------------------------------------------------------------------
   Start
   ------------------------------------------------------------------ */
async function load(first = true) {
  try {
    const data = await api("GET", "state");
    OPT = { ...OPT, ...data.options };
    state.refs = data.images.map(fromImage);
    state.clips = data.clips.filter((c) => !deleted.has(c.id));
    if (first) {
      driveStatus();
      loadRefmods();
      loadFaces();
      loadMotionStatus().then(render);
      const ui = data.ui || {};
      Object.assign(state.videoSettings, ui.settings || {});
      Object.assign(state.imageSettings, ui.imageSettings || {});
      Object.assign(state.kleinSettings, ui.kleinSettings || {});
      Object.assign(state.prunaSettings, ui.prunaSettings || {});
      if (!OPT.pruna.enhance) state.prunaSettings.enhance = false;
      state.prunaMode = ui.prunaMode === "sheet" ? "sheet" : "image";
      if (ui.prunaSheet && Array.isArray(ui.prunaSheet.views) && ui.prunaSheet.views.length) Object.assign(state.prunaSheet, ui.prunaSheet);
      state.imageEdit = !!ui.imageEdit && !!OPT.image.edit?.available;
      // Settings saved earlier must still suit what is installed, in every tab.
      for (const tab of ["video", "image", "klein", "pruna"]) {
        state.tab = tab;
        const T = tabOpt(), s = state.settings;
        if (T.models.length && !T.models.some((m) => m.name === s.model)) s.model = T.models[0].name;
        const m = modelOf(s.model), lo = m.minSteps || 1, every = m.stepSize || 1;
        if (s.steps < lo || s.steps > m.maxSteps || (s.steps - lo) % every) s.steps = m.steps;
        const lim = OPT.videoMP || { min: 0.1, max: 2 };
        if (tab === "video" ? !(s.res >= lim.min && s.res <= lim.max) : !T.megapixels.includes(s.res)) s.res = tab === "video" ? 0.6 : 1.0;
        if (!T.aspects[s.aspect]) s.aspect = tab === "video" ? "16:9" : "1:1";
        if (tab !== "video" && !(s.cfg >= OPT.image.cfg.min && s.cfg <= OPT.image.cfg.max)) s.cfg = m.cfg;
        if (tab === "pruna") {
          const pm = modelOf(s.model);
          if (!prunaMPs(pm).includes(s.sheetRes)) s.sheetRes = prunaFit(pm, 8.0);
          if (s.size !== "match" && !T.aspects[s.size]) s.size = "match";
          if (!prunaMPs(pm).includes(s.res)) s.res = prunaFit(pm, 4.0);
        }
        if (tab === "klein") {
          if (s.size !== "match" && !T.aspects[s.size]) s.size = "match";
          if (!m.base) s.cfg = m.cfg;   // Distilled renders without guidance
        }
      }
      // The first time the app opens in this browser tab, the Video tab starts on HyperFlow (when
      // installed), 5 seconds and 0.6 MP. After that (a reload, closing and opening it again) it keeps
      // the settings you had.
      let opened = false;
      try { opened = sessionStorage.getItem("mmvision.opened") === "1"; sessionStorage.setItem("mmvision.opened", "1"); } catch {}
      if (!opened) {
        const v = state.videoSettings, flow = OPT.models.find((m) => m.id === "hyperflow");
        if (flow) { v.model = flow.name; v.steps = flow.steps; }
        v.len = 5;
        v.res = 0.6;
      }
      EDIT.load(data.edit);
      state.tab = "video";   // the app always opens on the Video tab
      state.drafts = { video: ui.prompt || "", image: ui.imagePrompt || "", klein: ui.kleinPrompt || "", pruna: ui.prunaPrompt || "", prunaSheet: ui.prunaSheetPrompt || "" };
      const keep = (list, n) => Array.from({ length: n }, (_, i) => { const id = (list || [])[i]; return id && state.refs.some((r) => r.id === id) ? id : null; });
      state.prunaSlots = keep(ui.prunaSlots, 9);
      state.prunaSheetSlots = keep(ui.prunaSheetSlots, 4);
      state.kleinSwapSlots = keep(ui.kleinSwapSlots, 5);
      state.kleinSwapMask = ui.kleinSwapMask && ui.kleinSwapMask.mask ? ui.kleinSwapMask : null;
      state.kleinMode = ui.kleinMode === "swap" ? "swap" : "edit";
      state.drafts.kleinSwap = ui.kleinSwapPrompt || "";
      state.kleinSlots = [0, 1, 2, 3].map((i) => { const id = (ui.kleinSlots || [])[i]; return id && state.refs.some((r) => r.id === id) ? id : null; });
      state.imageEditSlot = [state.refs.some((r) => r.id === ui.imageEditSource) ? ui.imageEditSource : null];
      state.negative = typeof ui.imageNegative === "string" ? ui.imageNegative : "";
      state.negativeOpen = !!ui.negativeOpen;
      // A prompt bar saved by the subjects version (kept on the saved/video-subjects branch) has its pictures in
      // ui.subjects: they come back into the strip.
      const fromSubjects = (ui.subjects || []).flatMap((x) => x.refs || []);
      state.attached = [...new Set([...(ui.attached || []), ...fromSubjects])].filter((id) => state.refs.some((r) => r.id === id));
      state.motion = ui.motion && motionSource(ui.motion) ? ui.motion : null;
      state.motionMethod = ["guide", "swap"].includes(ui.motionMethod) ? ui.motionMethod : "tara";
      state.motionPeople = state.motion && Array.isArray(ui.motionPeople) ? ui.motionPeople : [];
      state.motionExtra = typeof ui.motionExtra === "string" ? ui.motionExtra : "";
      state.perRow = ui.perRow || 3;
      state.prompt = state.drafts[state.tab];
      state.caret = state.prompt.length;
      state.mentionClosed = true;
    }
    state.ready = true;
    state.loadError = null;
    shownClip = null;
    render();
  } catch (error) {
    state.loadError = error.message;
    if (!state.ready) renderGrid();
    clearTimeout(load.retry);
    load.retry = setTimeout(() => load(first), 4000);
  }
}


// Hovering an image in the prompt bar shows it bigger, above it.
const peek = document.createElement("div");
peek.className = "ref-peek";
peek.hidden = true;
peek.innerHTML = '<img alt="">';
document.body.appendChild(peek);
const hidePeek = () => { peek.hidden = true; };
document.addEventListener("mouseover", (e) => {
  const pic = e.target.closest?.(".ref-pic");
  if (!pic || e.target.closest(".ref-x") || e.buttons) { if (!e.target.closest?.(".ref-peek")) hidePeek(); return; }
  const src = pic.querySelector("img")?.src;
  if (!src) return;
  const img = peek.firstChild;
  const place = () => {
    const box = pic.getBoundingClientRect();
    const ratio = img.naturalWidth && img.naturalHeight ? img.naturalWidth / img.naturalHeight : 1.5;
    // Klein's slots get a big view (up to 560 px), below the top bar.
    const big = pic.classList.contains("slot-pic");
    const h = Math.min(big ? 560 : 300, box.top - (big ? belowTop() + 12 : 24)), w = Math.min(big ? 760 : 420, h * ratio, innerWidth - 24);
    peek.style.width = w + "px"; peek.style.height = w / ratio + "px";
    peek.style.left = Math.max(12, Math.min(innerWidth - w - 12, box.left + box.width / 2 - w / 2)) + "px";
    peek.style.top = box.top - w / ratio - 12 + "px";
    peek.hidden = false;
  };
  if (img.getAttribute("src") !== src) { img.onload = place; img.src = src; if (img.complete && img.naturalWidth) place(); }
  else place();
});
document.addEventListener("mouseout", (e) => { if (e.target.closest?.(".ref-pic") && !e.relatedTarget?.closest?.(".ref-pic")) hidePeek(); });
document.addEventListener("dragstart", hidePeek, true);
document.addEventListener("pointerdown", hidePeek, true);

// The prompt box's height. By itself it grows with the text, up to half the window, and longer text
// scrolls. Dragging the handle in its top-right corner sets a height of your own (up makes it taller)
// and a width (sideways; it stays centred), smaller than the text if you like. They last for the session.
const PROMPT_SIZE = "mmvision.prompt-size";
const PROMPT_MIN = 64;   // two lines of text
let promptSize = null;   // {width, height} you dragged to, or null
try { promptSize = JSON.parse(sessionStorage.getItem(PROMPT_SIZE) || "null"); } catch {}
function fitPrompt() {
  const root = document.documentElement.style, mirror = $("mirror");
  // Measuring at the smallest box makes the browser scroll the text (to the caret, often the end):
  // keep where it was scrolled to.
  const scrolled = ta.scrollTop;
  if (promptSize?.width) root.setProperty("--composer-width", promptSize.width + "px");
  let height = promptSize?.height;
  if (!height) {
    root.setProperty("--prompt-box", PROMPT_MIN + "px");   // measure the text at the smallest box
    height = Math.min(Math.max(PROMPT_MIN, mirror.scrollHeight), Math.round(innerHeight * 0.5));
  }
  root.setProperty("--prompt-box", height + "px");
  ta.scrollTop = scrolled;
  mirror.scrollTop = ta.scrollTop;
}
ta.addEventListener("scroll", () => { $("mirror").scrollTop = ta.scrollTop; });
addEventListener("resize", () => fitPrompt());
(function promptResizer() {
  const handle = document.createElement("div");
  handle.className = "prompt-resize";
  handle.title = "Drag to resize the prompt. Double-click to let it fit the text again.";
  handle.setAttribute("aria-hidden", "true");
  handle.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1 1h10v10M5 1h6v6"/></svg>';
  document.querySelector(".composer").appendChild(handle);
  handle.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const inner = document.querySelector(".dock-inner");
    const start = { x: e.clientX, y: e.clientY, w: inner.offsetWidth, h: document.querySelector(".prompt-wrap").offsetHeight };
    let moved = false;
    const move = (ev) => {
      if (!moved && Math.abs(ev.clientX - start.x) + Math.abs(ev.clientY - start.y) < 3) return;
      moved = true;
      promptSize = {
        width: Math.round(Math.max(520, Math.min(innerWidth - 64, start.w + 2 * (ev.clientX - start.x)))),
        height: Math.round(Math.max(38, Math.min(innerHeight * 0.75, start.h - (ev.clientY - start.y)))),
      };
      fitPrompt();
      if (state.settingsOpen) renderSettings();
    };
    const up = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", up);
      handle.removeEventListener("pointercancel", up);
      if (moved) try { sessionStorage.setItem(PROMPT_SIZE, JSON.stringify(promptSize)); } catch {}
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", up);
    handle.addEventListener("pointercancel", up);
  });
  // Double-click: back to fitting the text (the width you set stays).
  handle.addEventListener("dblclick", () => {
    if (promptSize) delete promptSize.height;
    try { sessionStorage.setItem(PROMPT_SIZE, JSON.stringify(promptSize)); } catch {}
    fitPrompt();
  });
  // A longer prompt makes the dock taller too, so the gallery keeps room below its last row.
  new ResizeObserver(() => document.documentElement.style.setProperty("--dock-space", $("dock").offsetHeight + 60 + "px")).observe($("dock"));
})();


// With Add image open, a click on the empty part of the image strip (right of Add image) closes it.
$("refs").addEventListener("click", (e) => {
  if (state.library.open && !e.target.closest("button, input, label, .ref-pic, .slot-empty, .add-ref")) ACT.closePopups();
});

$("negative").addEventListener("input", (e) => { state.negative = e.target.value; renderNegative(); saveUi(); });

// Start. Last, so everything above is defined.
// Only show the Close button when the app runs inside ComfyUI's page.
// The gear menu: its only item, Exit, goes back to ComfyUI, so it shows only inside ComfyUI.
// The gear menu: Models, and Exit (which goes back to ComfyUI, so only inside ComfyUI).
if (window.parent !== window) $("exitItem").hidden = false;
render();
load();
connect();
