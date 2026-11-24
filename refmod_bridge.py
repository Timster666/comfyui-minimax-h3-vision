"""RefMods for the Video tab, with Luisa's ComfyUI-MiniMaxH3Mod pack (https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).

A RefMod is a reference saved as a small file: pictures of a person (or clothes, a style...), or video clips of a style,
a look or a motion, encoded once with the H3 video VAE, no training. A render can then carry up to eight, each with a strength, on top of its prompt and pictures:
"Load H3 RefMods" reads them and "Apply H3 RefMod" adds them to the conditioning H3 samples with. They have no tag in
the prompt: the prompt describes who is there.

Everything here is optional: without the pack, Vision offers no RefMods and renders as before.
"""

import asyncio
import logging
import os
import re
import time
import uuid

from . import comfy_link, gdrive

LOG = logging.getLogger("minimax_flow")
LOADER, APPLY, EXTRACT = "MiniMaxH3RefModsLoader", "MiniMaxH3RefModApply", "MiniMaxH3RefModExtract"
SLOTS = 8
LOADER_NODE, APPLY_NODE = "172", "173"
# How a RefMod keeps its pictures: pooled down (quick to render with), or the whole encode (more detail, slower).
MODES = {"compressed": {"mode": "Compressed Reference", "extraction_preset": "manual", "max_tokens": 5120},
         "full": {"mode": "Full Reference", "extraction_preset": "identity_encode", "max_tokens": 8192}}
CONCEPTS = ["identity", "clothing", "style", "background", "pose_motion", "generic"]
PICTURES_MAX = 16
VIDEOS_MAX = 8
# Video clips: read at 24 fps, small (the pack brings them down to ref_resolution anyway), and only their first seconds.
VIDEO_MEGAPIXELS = 0.4
SECONDS = (2, 10)
# The pack's presets for what a video RefMod holds (Compressed only): a style or a look pooled to a small grid; a
# motion kept as an ordered sequence of frames.
VIDEO_PRESETS = {"style": "style_experimental", "pose_motion": "motion_sequence"}


def installed():
    import nodes
    return all(n in nodes.NODE_CLASS_MAPPINGS for n in (LOADER, APPLY, EXTRACT))


def files():
    """The RefMods the pack's loader offers (names without .safetensors), or None without the pack."""
    import nodes
    if not installed():
        return None
    try:
        loader = nodes.NODE_CLASS_MAPPINGS[LOADER]
        options = loader.INPUT_TYPES()["required"]["mod_1"][0]
    except Exception:
        LOG.exception("MiniMax H3 Vision couldn't read the RefMod list")
        return []
    return sorted((o for o in options if isinstance(o, str) and o not in ("", "None", "(none)")), key=str.lower)


def clean(raw):
    """A render's RefMods as kept with the clip: [{"file", "strength"}], at most eight, each one that exists. ValueError
    for one that doesn't."""
    if not raw:
        return []
    have = files()
    if have is None:
        raise ValueError("RefMods need the ComfyUI-MiniMaxH3Mod node pack, which isn't installed.")
    out = []
    for mod in list(raw)[:SLOTS]:
        name = str((mod or {}).get("file") or "")
        if name not in have:
            raise ValueError("There's no RefMod called %s (any more)." % name)
        strength = min(1.0, max(0.0, float((mod or {}).get("strength", 1.0))))
        if strength > 0 and name not in [m["file"] for m in out]:
            out.append({"file": name, "strength": round(strength, 2)})
    return out


def add_to(graph, mods, conditioning_of=("13", "conditioning")):
    """Put the RefMods into a video graph: loaded, and applied to what the guider conditions on."""
    if not mods:
        return graph
    loader = {"show_info": False}
    for slot in range(1, SLOTS + 1):
        mod = mods[slot - 1] if slot <= len(mods) else None
        loader.update({"mod_%d" % slot: mod["file"] if mod else "(none)", "strength_%d" % slot: mod["strength"] if mod else 1.0,
                       "copies_%d" % slot: 1})
    graph[LOADER_NODE] = {"class_type": LOADER, "inputs": loader}
    node, key = conditioning_of
    graph[APPLY_NODE] = {"class_type": APPLY, "inputs": {
        "conditioning": graph[node]["inputs"][key], "mods": [LOADER_NODE, 0], "override": False, "retention": 1.0,
        "curve_direction": "constant", "scramble_seed": -1, "curve_shape": "linear", "curve_value": 1.0}}
    graph[node]["inputs"][key] = [APPLY_NODE, 0]
    return graph


def safe_name(text):
    name = re.sub(r"[^A-Za-z0-9_-]", "", str(text or "").strip().replace(" ", "_"))[:48]
    if not name:
        raise ValueError("Give the RefMod a name with letters or numbers.")
    return name


def make_graph(name, pictures, vae, mode="compressed", concept="identity", description="", videos=(), seconds=5,
               merge=False, motion_only=False):
    """The graph that makes a RefMod from pictures and video clips (paths ComfyUI can load; the videos' inside the
    output folder) with the pack's Create H3 RefMod."""
    if not pictures and not videos:
        raise ValueError("Pick the pictures or video clips to make the RefMod from.")
    if mode not in MODES:
        raise ValueError("Unknown RefMod mode.")
    if concept not in CONCEPTS:
        raise ValueError("Unknown kind of RefMod.")
    settings = MODES[mode]
    graph = {"1": {"class_type": "VAELoader", "inputs": {"vae_name": vae}}}
    extract = {"name": name, "mode": settings["mode"], "concept_type": concept, "background_retention": 0.0,
               "ref_resolution": 1024, "pool_h": 16, "pool_w": 16, "latent_frames": 16, "identity": 500, "merge": False,
               "motion_only": False, "multiplier": 1, "max_tokens": settings["max_tokens"],
               "description": str(description or "").strip()[:300], "save": True, "vae": ["1", 0],
               "extraction_preset": settings["extraction_preset"], "budget_policy": "truncate", "image_refs": "stack"}
    for n, path in enumerate(pictures[:PICTURES_MAX]):
        node = str(10 + n)
        graph[node] = {"class_type": "MMVisionLoadStill", "inputs": {"file": path}}
        extract["refs_image.ref_image_%d" % n] = [node, 0]
    if videos:
        seconds = min(SECONDS[1], max(SECONDS[0], float(seconds or 5)))
        for n, path in enumerate(list(videos)[:VIDEOS_MAX]):
            node = str(40 + n)
            graph[node] = {"class_type": "MMVisionLoadGuide", "inputs": {"file": path, "megapixels": VIDEO_MEGAPIXELS,
                                                                         "max_frames": int(seconds * 24) + 5}}
            extract["refs_video.ref_video_%d" % n] = [node, 0]
        # A clip's frames: 512 px on the short side is plenty for a look or a motion, and keeps the encode quick.
        extract["ref_resolution"] = 512
        if mode == "compressed":
            extract["extraction_preset"] = VIDEO_PRESETS.get(concept, "manual")
        else:
            # Full keeps the clip's encode, frames up to the token budget (the pack drops near-duplicates first).
            extract["extraction_preset"] = "manual"
            extract["latent_frames"] = 39
    motion_only = bool(motion_only) and bool(videos) and concept == "pose_motion"
    if merge or motion_only:
        # Merge: one shared reference refined against every picture and clip at once, so it holds what they have in
        # common at one reference's size. Only the movement: clips turned into frame-to-frame differences first, so
        # how things look stays out. The pack does both in Compressed only, and its presets switch them off, so the
        # presets' settings are set by hand: a style's small 8x8 grid, 16x16 otherwise.
        if mode != "compressed":
            raise ValueError("Merge and Only the movement work with Compressed only.")
        if merge and len(pictures) + len(videos) < 2:
            raise ValueError("Merge needs at least two pictures or clips.")
        small = concept == "style"
        extract.update({"extraction_preset": "manual", "pool_h": 8 if small else 16, "pool_w": 8 if small else 16,
                        "identity": 150 if small else 300, "merge": bool(merge), "motion_only": motion_only})
    graph["2"] = {"class_type": EXTRACT, "inputs": extract}
    return graph


def refmods_dir():
    """Where the pack saves and loads RefMods (models/refmods)."""
    import folder_paths
    try:
        roots = folder_paths.get_folder_paths("refmods")
    except Exception:
        roots = []
    return roots[0] if roots else os.path.join(folder_paths.models_dir, "refmods")


class DriveCopy:
    """RefMods kept in the Google Drive folder too, in a refmods folder there: the ones on Drive that this pod lacks are
    fetched, and the ones made here are uploaded. Nothing is deleted on either side. Runs on the first look at the
    RefMods after ComfyUI starts, after each RefMod is made, and when asked."""

    def __init__(self):
        self.state = {"state": "off", "error": None, "at": None, "fetched": 0}
        self.task = None
        self.done_once = False
        self.on_change = None

    def on(self):
        return gdrive.problem() is None

    def view(self):
        if not self.on():
            return {"state": "off", "error": None, "at": None, "fetched": 0}
        return self.state if self.state["state"] != "off" else {**self.state, "state": "idle"}

    def start(self):
        if self.on() and (not self.task or self.task.done()):
            # Syncing from now, so the page that asked keeps looking until it's done.
            self.state = {**self.state, "state": "syncing", "error": None}
            self.task = asyncio.ensure_future(self.sync())
        return self.view()

    async def sync(self):
        folder = refmods_dir()
        os.makedirs(folder, exist_ok=True)
        before = {f for f in os.listdir(folder) if f.endswith(".safetensors")}
        self.state = {**self.state, "state": "syncing", "error": None}
        try:
            only = ["--include", "*.safetensors", "--ignore-existing"]
            # The refmods folder on Drive is made the first time; copying from it before it exists fails.
            await gdrive.DRIVE.run("mkdir", gdrive.remote("refmods"))
            await gdrive.DRIVE.run("copy", folder, gdrive.remote("refmods"), *only)
            await gdrive.DRIVE.run("copy", gdrive.remote("refmods"), folder, *only)
            after = {f for f in os.listdir(folder) if f.endswith(".safetensors")}
            fetched = len(after - before)
            import json
            on_drive = [f for f in json.loads(await gdrive.DRIVE.run("lsjson", gdrive.remote("refmods"), "--files-only") or "[]")
                        if f["Name"].endswith(".safetensors")]
            self.state = {"state": "synced", "error": None, "at": time.time(), "fetched": fetched,
                          "here": len(after), "drive": len(on_drive), "folder": folder}
            if fetched and self.on_change:
                self.on_change()
        except Exception as error:
            LOG.warning("MiniMax H3 Vision couldn't copy RefMods to or from Google Drive: %s", error)
            self.state = {"state": "error", "error": str(error) or type(error).__name__, "at": None, "fetched": 0}
        self.done_once = True


DRIVE_COPY = DriveCopy()


class Maker:
    """RefMods being made: queued in ComfyUI like a render, watched until the file is there."""

    def __init__(self):
        self.jobs = {}
        self.on_done = None

    def status(self):
        now = time.time()
        for jid in [j for j, job in self.jobs.items() if job["state"] != "making" and now - job.get("ended", now) > 600]:
            del self.jobs[jid]
        return sorted(self.jobs.values(), key=lambda j: j["at"], reverse=True)

    async def make(self, name, graph):
        prompt_id = await comfy_link.queue_prompt(graph)
        jid = uuid.uuid4().hex[:8]
        self.jobs[jid] = {"id": jid, "name": name, "state": "making", "error": None, "at": time.time(), "prompt_id": prompt_id}
        asyncio.ensure_future(self._watch(jid))
        return self.jobs[jid]

    async def _watch(self, jid):
        job = self.jobs[jid]
        for _ in range(1800):
            await asyncio.sleep(2)
            entry = comfy_link.history(job["prompt_id"])
            if entry is None:
                continue
            status = entry.get("status") or {}
            kinds = [kind for kind, _ in status.get("messages") or []]
            if status.get("status_str") == "error" or "execution_error" in kinds or "execution_interrupted" in kinds:
                job.update(state="failed", error=comfy_link.failure(entry))
                break
            if not status.get("completed"):
                continue
            # Done: the file can take a moment to be listed.
            for _ in range(10):
                if job["name"] in (files() or []):
                    break
                await asyncio.sleep(1)
            if job["name"] in (files() or []):
                job.update(state="done")
            else:
                job.update(state="failed", error="ComfyUI finished, but the RefMod %s isn't in the list." % job["name"])
            break
        else:
            job.update(state="failed", error="It took too long.")
        job["ended"] = time.time()
        if job["state"] == "done":
            DRIVE_COPY.start()   # onto Google Drive too
        if self.on_done:
            self.on_done(job)


MAKER = Maker()
