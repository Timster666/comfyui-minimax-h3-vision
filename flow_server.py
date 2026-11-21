"""The server side of MiniMax H3 Vision: its page, its API, and the renderer.

Everything lives in ComfyUI's output folder under minimax_flow/:
    project.json   images, clips and the last-used settings
    images/        uploaded reference images
    clips/         finished videos
    stills/        finished images (the Image tab)
Files there are served by ComfyUI's own /view endpoint.

Videos are MiniMax H3 graphs built here (h3_graph.py, with this pack's nodes in vision_nodes.py), and
images are MOSS-Image graphs (moss_bridge.py). Both run in ComfyUI's own queue. A tab whose models
aren't installed says which files it needs.
"""

import asyncio
import fractions
import contextvars
import base64
import json
import logging
import math
import os
import random
import re
import shutil
import time
import uuid

from aiohttp import web

from . import clip_info, comfy_link, edit_media, edit_project, face_refine, gdrive, swap, prompt_lib, refmod_bridge, h3_graph, klein_bridge, lora_in, model_store, moss_bridge, ph3_bridge

HERE = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(HERE, "app")
ROOT = "minimax_flow"
EVENT = "mmflow.clip"
LOG = logging.getLogger("minimax_flow")

ASPECTS = {"16:9": [16, 9], "9:16": [9, 16], "1:1": [1, 1], "4:3": [4, 3], "3:4": [3, 4], "21:9": [21, 9]}

# The Image tab (MOSS-Image): any aspect ratio, sides in steps of 16, about 0.25 to 4 MP, best at 1 to 2 MP.
IMAGE_ASPECTS = {"1:1": [1, 1], "4:5": [4, 5], "3:4": [3, 4], "2:3": [2, 3], "9:16": [9, 16],
                 "16:9": [16, 9], "3:2": [3, 2], "4:3": [4, 3], "5:4": [5, 4], "21:9": [21, 9]}
IMAGE_MEGAPIXELS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
IMAGE_RECOMMENDED_MP = [1.0, 1.5, 2.0]
CFG = {"min": 1, "max": 10, "step": 0.5}
KLEIN_MEGAPIXELS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
KLEIN_RECOMMENDED_MP = [1.0, 1.5, 2.0]
# The Pruna tab (P-H3-Image): sides in steps of 32, 1 to 8 MP; 8 MP is the model's own size.
PH3_ASPECTS = {"1:1": [1, 1], "4:3": [4, 3], "3:2": [3, 2], "16:9": [16, 9], "21:9": [21, 9],
               "3:4": [3, 4], "2:3": [2, 3], "9:16": [9, 16]}
MEGAPIXELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
VIDEO_MP = {"min": 0.1, "max": 2.0}     # any size in between can be typed in, in hundredths
# H3-Base's own size (MiniMax: 1344x768 at 16:9, sides in multiples of 32, at most 768x1344 pixels). "1.0 MP" is
# this, as large as the limit allows in each shape, rather than 1024x1024 pixels, which is just over it.
H3_LIMIT = 1344 * 768
H3_NATIVE = {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (992, 992), "4:3": (1152, 864), "3:4": (864, 1152),
             "21:9": (1536, 672)}
LENGTH = {"min": 2, "max": 20, "fps": 24}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
MENTION_RE = re.compile(r"@([A-Za-z0-9_]+)")

# The Edit tab: one track of pieces, each a part of a finished video: {id, clip, in, out} in seconds.
DEFAULT_EDIT = {"name": "Untitled edit", "pieces": [], "ratio": "16:9", "muted": False}
EXPORT_EVENT = "mmvision_export"
_exports = {}   # export id -> {"cancel": bool}

DEFAULT_UI = {
    "motion": None,   # the Video tab's Motion from: the id of the video that gives the motion, or None
    "motionMethod": "tara",   # and how: "tara" (Motion Transfer TARA v1.0) or "guide" (Motion Transfer v1.0)
    "settings": {"model": "HyperFlow H3 Flash", "steps": 6, "res": 0.6, "aspect": "16:9", "len": 5,
                 "seedMode": "random", "seed": 48213, "loras": [], "realism": {"on": True, "strength": 1.2}},
    "imageSettings": {"model": "MOSS-Image SFT (INT8)", "steps": 50, "cfg": 3.0, "res": 1.0, "aspect": "1:1",
                      "seedMode": "random", "seed": 48213, "loras": []},
    "attached": [],
    "perRow": 3,
    "prompt": "",
    "imagePrompt": "",
    # The Klein tab: its settings, prompt and four slots (image ids from the Add image library, or None).
    # size is "match" (image 1's shape) or an aspect ratio.
    "kleinSettings": {"model": "Klein 9B Distilled (BF16)", "steps": 4, "cfg": 1.0, "res": 1.0, "size": "match", "aspect": "1:1",
                      "seedMode": "random", "seed": 48213, "loras": [],
                      # Face swap: its size, the BFS LoRA, the sampler with a head mask, Refine face, the mask's edge.
                      "swapRes": 2.0, "swapLora": "", "swapSampler": "standard", "lanSteps": 20, "refine": False,
                      "refineStrength": 0.5, "maskGrow": 8, "maskFeather": 12},
    "kleinPrompt": "",
    "kleinSlots": [None, None, None, None],
    "kleinMode": "edit",              # Klein: Edit, or Face swap (body and face, with the BFS LoRA)
    "kleinSwapSlots": [None] * 5,
    "kleinSwapMask": None,            # the head mask painted on the body picture: {"image": id, "mask": file}
    "kleinSwapPrompt": "",
    # The Pruna tab: Image (up to nine slots, picture 1-9) or Character sheet (the character and up to three
    # helper pictures with a role each, and the views to draw).
    "prunaSettings": {"model": ph3_bridge.LORA_ROUTE, "steps": 8, "res": 4.0, "sheetRes": 8.0, "size": "match", "aspect": "1:1",
                      "seedMode": "random", "seed": 48213, "loras": [], "enhance": False},
    "prunaPrompt": "",
    "prunaSheetPrompt": "",
    "prunaMode": "image",
    "prunaSlots": [None] * 9,
    "prunaSheetSlots": [None] * 4,
    "prunaSheet": {"preset": "standard", "views": ["head_front", "head_side", "body_front", "body_side", "body_back"],
                   "roles": ["character", "face", "outfit", "detail"], "bg": "white"},
    "imageNegative": moss_bridge.NEGATIVE,   # the Image tab's negative prompt box starts with this text
    "imageEdit": False,                      # the Image tab's Edit switch (one picture, with the edit LoRA)
    "imageEditSource": None,                 # the picture to edit: an Add image library id
    "negativeOpen": False,
    "negativeShown": True,
}

_lock = asyncio.Lock()


# ---------------------------------------------------------------- storage

def output_dir():
    import folder_paths
    return folder_paths.get_output_directory()


def root_dir(*parts):
    path = os.path.join(output_dir(), ROOT, *parts)
    os.makedirs(path if not os.path.splitext(path)[1] else os.path.dirname(path), exist_ok=True)
    return path


def project_file():
    return os.path.join(root_dir(), "project.json")


def sampling_choices():
    """ComfyUI's samplers and schedulers, as its KSampler lists them."""
    try:
        import comfy.samplers
        return list(comfy.samplers.KSampler.SAMPLERS), list(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:
        return (["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde", "res_multistep", "uni_pc"],
                ["simple", "normal", "karras", "exponential", "sgm_uniform", "beta"])


def pick_sampling(s, model, video):
    """The sampler and scheduler a render asked for: None means the model's own."""
    samplers, schedulers = sampling_choices()
    if video:
        schedulers = schedulers + [h3_graph.BETA57]
    sampler = s.get("sampler") or None
    scheduler = s.get("scheduler") or None
    if sampler and sampler not in samplers:
        raise web.HTTPBadRequest(text="ComfyUI has no sampler called %s." % sampler)
    if scheduler and (scheduler not in schedulers or model.get("scheduler") == h3_graph.HYPERFLOW_SCHEDULE):
        raise web.HTTPBadRequest(text="%s can't use the %s scheduler." % (model["name"], scheduler))
    return sampler, scheduler


def video_models():
    """The installed MiniMax H3 models, and why there are none when there are none."""
    try:
        models = h3_graph.models()
        return models, (None if models else h3_graph.why_none())
    except Exception as error:
        # Never let this break the app: say what went wrong.
        LOG.exception("MiniMax H3 Vision couldn't list its video models")
        return [], "MiniMax H3 Vision couldn't list its video models: %s: %s" % (type(error).__name__, error)


def image_models():
    """The installed MOSS-Image models, and why there are none when there are none."""
    try:
        return moss_bridge.models(), None
    except moss_bridge.Missing as error:
        return [], str(error)
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't check for MOSS-Image")
        return [], "MiniMax H3 Vision couldn't check for MOSS-Image: %s: %s" % (type(error).__name__, error)


def klein_models():
    """The installed Klein 9B models, and why there are none when there are none."""
    try:
        return klein_bridge.models(), None
    except klein_bridge.Missing as error:
        return [], str(error)
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't check for Klein")
        return [], "MiniMax H3 Vision couldn't check for Klein: %s: %s" % (type(error).__name__, error)


def pruna_models():
    """The installed ways to run P-H3-Image, and why there are none when there are none."""
    try:
        return ph3_bridge.models(), None
    except ph3_bridge.Missing as error:
        return [], str(error)
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't check for P-H3-Image")
        return [], "MiniMax H3 Vision couldn't check for P-H3-Image: %s: %s" % (type(error).__name__, error)


def read_project():
    try:
        with open(project_file(), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data.setdefault("images", [])
    data.setdefault("clips", [])
    data.setdefault("edit", dict(DEFAULT_EDIT))
    ui = data.setdefault("ui", {})
    # Before the negative prompt box showed its text, an empty box meant "the recommended one".
    if "negativeShown" not in ui and not ui.get("imageNegative"):
        ui["imageNegative"] = moss_bridge.NEGATIVE
    for key, value in DEFAULT_UI.items():
        ui.setdefault(key, json.loads(json.dumps(value)))
    return data


def write_project(data):
    path = project_file()
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(temp, path)


def broadcast(clip):
    from server import PromptServer
    PromptServer.instance.send_sync(EVENT, {"clip": clip}, None)


def dims(aspect, mp, grid=32, table=None):
    """Width and height for an aspect ratio at a size in megapixels (1 MP = 1024x1024).
    H3 wants sides in steps of 32, MOSS-Image in steps of 16."""
    a, b = (table or ASPECTS)[aspect]
    pixels = mp * 1024 * 1024
    w = max(grid, round(math.sqrt(pixels * a / b) / grid) * grid)
    h = max(grid, round(math.sqrt(pixels * b / a) / grid) * grid)
    return w, h


def video_dims(aspect, mp):
    """MiniMax H3's width and height: 1.0 MP is H3's own size; up to it, sizes stay within its pixel limit; and a
    size that would come out more than 3% the wrong shape (small ones, rounded to 32) is picked for its shape.
    app.js has the same rules."""
    if abs(mp - 1.0) < 1e-6:
        return H3_NATIVE[aspect]
    a, b = ASPECTS[aspect]
    ratio, pixels, g = a / b, mp * 1024 * 1024, 32
    cap = H3_LIMIT if mp <= 1.0 else None
    w = max(g, round(math.sqrt(pixels * ratio) / g) * g)
    h = max(g, round(math.sqrt(pixels / ratio) / g) * g)
    if abs(w / h / ratio - 1) <= 0.03 and (cap is None or w * h <= cap):
        return w, h
    best = None
    for h in range(g, 4096, g):
        for w in {max(g, math.floor(h * ratio / g) * g), math.ceil(h * ratio / g) * g}:
            if cap and w * h > cap:
                continue
            area, shape = abs(w * h / pixels - 1), abs(w / h / ratio - 1)
            if area > 0.12:
                continue
            score = shape + 0.35 * area
            if best is None or score < best[0] - 1e-12:
                best = (score, w, h)
    return best[1], best[2]


def recover(data):
    """Renders that were running when ComfyUI stopped will never finish."""
    for clip in data["clips"]:
        if clip.get("status") in ("queued", "rendering"):
            clip["status"] = "failed"
            clip["error"] = "ComfyUI was restarted during this render."


# ---------------------------------------------------------------- page and API

async def page(request):
    raise web.HTTPFound("/mmflow/app/index.html?v=%d" % int(time.time()))


async def get_state(request):
    async with _lock:
        data = read_project()
        if not getattr(get_state, "recovered", False):
            recover(data)
            write_project(data)
            get_state.recovered = True
    loras = lora_list()
    models, why = video_models()
    imodels, iwhy = image_models()
    kmodels, kwhy = klein_models()
    pmodels, pwhy = pruna_models()
    if any(c.get("status") in ("queued", "rendering") and c.get("prompt_id") for c in data["clips"]):
        ensure_watcher()
    return web.json_response({
        **data,
        "options": {"editSizes": EDIT_SIZES, "samplers": sampling_choices()[0], "schedulers": sampling_choices()[1],
                    "beta57": h3_graph.BETA57, "models": models, "aspects": ASPECTS, "megapixels": MEGAPIXELS, "videoMP": VIDEO_MP, "length": LENGTH,
                    "loras": loras, "missing": why, "root": ROOT,
                    "realism": dict(h3_graph.REALISM, file=h3_graph.realism_lora()),
                    "image": {"models": imodels, "aspects": IMAGE_ASPECTS, "megapixels": IMAGE_MEGAPIXELS,
                              "recommendedMP": IMAGE_RECOMMENDED_MP, "cfg": CFG,
                              # Edit (one picture, with the edit LoRA): its settings, and whether the LoRA is there.
                              "edit": dict(moss_bridge.EDIT, available=bool(moss_bridge.edit_lora()), lora=moss_bridge.EDIT_LORA),
                              "missing": iwhy},
                    "klein": {"models": kmodels, "aspects": IMAGE_ASPECTS, "megapixels": KLEIN_MEGAPIXELS,
                              "recommendedMP": KLEIN_RECOMMENDED_MP, "cfg": CFG, "slots": klein_bridge.SLOTS,
                              "swap": bool(klein_bridge.bfs_lora()), "swapPrompt": klein_bridge.SWAP_PROMPT,
                              "swapLoras": klein_bridge.bfs_loras(), "lanpaint": klein_bridge.lanpaint(),
                              "swapSlots": klein_bridge.SWAP_SLOTS,
                              "missing": kwhy},
                    "pruna": {"models": pmodels, "aspects": PH3_ASPECTS, "megapixels": ph3_bridge.MEGAPIXELS,
                              "slots": ph3_bridge.SLOTS, "steps": ph3_bridge.STEPS, "enhance": bool(ph3_bridge.upsampler()),
                              "missing": pwhy}},
    })


def lora_list(refresh=False):
    """The LoRA files in ComfyUI's loras folders. refresh: look again instead of using ComfyUI's cached list,
    which only notices a change in the top folder, not a file added in a subfolder."""
    try:
        import folder_paths
        if refresh:
            getattr(folder_paths, "filename_list_cache", {}).pop("loras", None)
        return sorted(folder_paths.get_filename_list("loras"), key=str.lower)
    except Exception:
        LOG.exception("MiniMax H3 Vision couldn't list the LoRAs")
        return []


async def get_loras(request):
    """The LoRAs and their sizes, for the Add LoRA window. ?refresh=1 looks in the folders again."""
    names = await asyncio.to_thread(lora_list, request.query.get("refresh") == "1")

    def sizes():
        import folder_paths
        out = {}
        for name in names:
            try:
                out[name] = os.path.getsize(folder_paths.get_full_path("loras", name))
            except Exception:
                pass
        return out
    try:
        found = await asyncio.to_thread(sizes)
    except Exception:
        found = {}
    return web.json_response({"loras": names, "sizes": found})


async def put_ui(request):
    body = await request.json()
    async with _lock:
        data = read_project()
        for key in DEFAULT_UI:
            if key in body:
                data["ui"][key] = body[key]
        write_project(data)
    return web.json_response({"ok": True})


# ---- The Models page: download model_catalog.json's groups in the background ----
def models_done(group_id):
    """A group finished downloading: tell the app, which reloads what is installed."""
    from server import PromptServer
    lora_list(refresh=True)
    PromptServer.instance.send_sync("mmflow.models", {"done": group_id}, None)


async def get_models(request):
    if model_store.STORE.sizes_file is None:
        model_store.STORE.load_sizes(root_dir())
        model_store.STORE.on_done = models_done
    return web.json_response(model_store.STORE.status())


# ---- The prompt builder's library (here, and in the Google Drive folder when it is set up) ----
def prompts_changed():
    from server import PromptServer
    PromptServer.instance.send_sync("mmflow.prompts", {}, None)


async def prompts(request):
    """GET: the library (fetched from Drive on the first read); PUT {lib}: the library after a change (merged with
    what is here); POST .../sync: fetch, merge and upload now."""
    lib = prompt_lib.LIBRARY
    lib.on_change = prompts_changed
    path = root_dir("prompt_library.json")
    if request.method == "GET":
        return web.json_response(await lib.get(path))
    if request.method == "PUT":
        body = await request.json()
        try:
            return web.json_response(await lib.put(path, body.get("lib")))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error))
    lib.load(path)
    if lib.drive_on():
        await lib.sync()
    return web.json_response(lib.view())


# ---- RefMods (with the ComfyUI-MiniMaxH3Mod pack) ----
def refmod_made(job):
    from server import PromptServer
    PromptServer.instance.send_sync("mmflow.refmods", {"job": {k: job[k] for k in ("id", "name", "state", "error")}}, None)


async def refmods(request):
    """GET: whether the pack is there, the RefMods, and those being made. POST {name, pictures, mode, concept, desc,
    replace}: make one from pictures in Add image."""
    maker = refmod_bridge.MAKER
    maker.on_done = refmod_made
    copy = refmod_bridge.DRIVE_COPY
    copy.on_change = lambda: refmod_made({"id": "", "name": "", "state": "fetched", "error": None})
    have = refmod_bridge.files()
    if request.method == "GET":
        # The first look after ComfyUI starts fetches the RefMods on Google Drive; ?sync=1 copies both ways now.
        if have is not None and (request.query.get("sync") == "1" or not copy.done_once):
            copy.start()
        return web.json_response({"installed": have is not None, "files": have or [], "jobs": maker.status(),
                                  "modes": list(refmod_bridge.MODES), "concepts": refmod_bridge.CONCEPTS,
                                  "drive": copy.view()})
    if have is None:
        raise web.HTTPBadRequest(text="Making a RefMod needs the ComfyUI-MiniMaxH3Mod node pack. On RunPod: set ENABLE_REFMOD=true.")
    body = await request.json()
    try:
        name = refmod_bridge.safe_name(body.get("name"))
        if name in have and not body.get("replace"):
            raise web.HTTPConflict(text="There's a RefMod called %s already." % name)
        async with _lock:
            by_id = {i["id"]: i for i in read_project()["images"]}
        pictures = ["%s/images/%s" % (ROOT, by_id[i]["file"]) for i in body.get("pictures") or [] if i in by_id]
        found, missing = h3_graph.encoders(h3_graph.available())
        if not found.get("vae"):
            raise ValueError("Making a RefMod needs MiniMax H3's video VAE (minimax_h3_video_vae_fp16.safetensors in models/vae).")
        graph = refmod_bridge.make_graph(name, pictures, found["vae"], body.get("mode") or "compressed",
                                         body.get("concept") or "identity", body.get("desc") or "")
        job = await maker.make(name, graph)
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    except web.HTTPException:
        raise
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't make a RefMod")
        raise web.HTTPBadRequest(text="ComfyUI refused it: %s" % error)
    return web.json_response({"job": job, "jobs": maker.status()})


async def models_token(request):
    """POST {token, remember}: check a Hugging Face token and use it for downloads; DELETE: forget it."""
    if request.method == "DELETE":
        model_store.forget_token()
    else:
        body = await request.json()
        try:
            await model_store.set_token(body.get("token"), bool(body.get("remember")))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error))
    return web.json_response(model_store.STORE.status())


async def models_action(request):
    group, action = request.match_info["id"], request.match_info["action"]
    store = model_store.STORE
    store.on_done = models_done
    try:
        if action == "download":
            store.download(group)
        elif action == "cancel":
            store.cancel(group)
        elif action == "delete":
            return web.json_response({**store.delete(group), **store.status()})
        else:
            raise web.HTTPNotFound()
    except KeyError:
        raise web.HTTPNotFound(text="No such model group.")
    except RuntimeError as error:
        raise web.HTTPBadRequest(text=str(error))
    return web.json_response(store.status())


# ---- LoRAs into the loras folder: uploaded in pieces, or fetched from a link ----
def lora_added(file):
    """A LoRA is in: the lists are read again, and the app is told."""
    from server import PromptServer
    lora_list(refresh=True)
    PromptServer.instance.send_sync("mmflow.loras", {"file": file}, None)


async def lora_upload(request):
    """POST {name, folder, size} starts an upload; PUT ?offset= sends a piece (the answer is where to go on from);
    POST .../finish {overwrite} checks the file and puts it in place; DELETE cancels."""
    inc = lora_in.INCOMING
    uid = request.match_info.get("id")
    try:
        if request.method == "POST" and not uid:
            body = await request.json()
            return web.json_response(inc.start_upload(body.get("name"), body.get("folder"), body.get("size", 0)))
        if request.method == "PUT":
            data = await request.read()
            return web.json_response({"offset": inc.write(uid, request.query.get("offset", "0"), data)})
        if request.method == "POST":
            body = await request.json()
            file = inc.finish(uid, bool(body.get("overwrite")))
            lora_added(file)
            return web.json_response({"file": file})
        inc.cancel_upload(uid)
        return web.json_response({})
    except KeyError:
        raise web.HTTPNotFound(text="That upload isn't going on any more: start it again.")
    except FileExistsError as error:
        raise web.HTTPConflict(text="%s is already there." % error)
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))


async def lora_links(request):
    """GET: the links being fetched; POST {url, folder}: fetch one; POST .../{id}/cancel."""
    inc = lora_in.INCOMING
    inc.on_done = lora_added
    if request.method == "GET":
        return web.json_response({"jobs": inc.status()})
    jid = request.match_info.get("id")
    if jid:
        inc.cancel_job(jid)
        return web.json_response({"jobs": inc.status()})
    body = await request.json()
    try:
        inc.link(body.get("url"), body.get("folder"))
    except ValueError as error:
        raise web.HTTPBadRequest(text=str(error))
    return web.json_response({"jobs": inc.status()})


async def upload_mask(request):
    """A head mask painted in the app (a PNG, white where the picture may change), kept in minimax_flow/masks."""
    body = await request.json()
    data = str(body.get("data") or "")
    if not data.startswith("data:image/png;base64,"):
        raise web.HTTPBadRequest(text="That isn't a PNG mask.")
    raw = base64.b64decode(data.split(",", 1)[1])
    if len(raw) > 40 * 1024 * 1024:
        raise web.HTTPBadRequest(text="That mask is too big.")
    name = "m" + uuid.uuid4().hex[:10] + ".png"
    with open(root_dir("masks", name), "wb") as f:
        f.write(raw)
    return web.json_response({"mask": name, "file": "%s/masks/%s" % (ROOT, name)})


async def upload_images(request):
    reader = await request.multipart()
    added = []
    async with _lock:
        data = read_project()
        taken = {i["name"] for i in data["images"]}
        while True:
            part = await reader.next()
            if part is None:
                break
            if not part.filename:
                continue
            ext = os.path.splitext(part.filename)[1].lower()
            if ext not in IMAGE_TYPES:
                continue
            image_id = "i" + uuid.uuid4().hex[:10]
            file = image_id + ext
            with open(root_dir("images", file), "wb") as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
            base = re.sub(r"[^a-z0-9_]+", "_", os.path.splitext(part.filename)[0].lower()).strip("_")[:16] or "image"
            name, n = base, 2
            while name in taken:
                name, n = "%s%d" % (base, n), n + 1
            taken.add(name)
            image = {"id": image_id, "name": name, "file": file, "added": time.time()}
            data["images"].append(image)
            added.append(image)
        write_project(data)
    return web.json_response({"images": added})


async def rename_image(request):
    image_id = request.match_info["id"]
    name = str((await request.json()).get("name", ""))
    if not NAME_RE.match(name):
        raise web.HTTPBadRequest(text="Names need letters, digits or _.")
    async with _lock:
        data = read_project()
        image = next((i for i in data["images"] if i["id"] == image_id), None)
        if not image:
            raise web.HTTPNotFound(text="No such image.")
        if any(i["name"] == name and i is not image for i in data["images"]):
            raise web.HTTPBadRequest(text="Another image already has that name.")
        image["name"] = name
        write_project(data)
    return web.json_response({"image": image})


# ---- "What the model gets": the same request as Generate, stopped just before it is queued ----
_PREVIEW = contextvars.ContextVar("mmflow_preview", default=False)


class _Preview(BaseException):
    """Raised instead of queueing in a preview (a BaseException, so the render paths' error handling lets it by)."""
    def __init__(self, graph):
        super().__init__()
        self.graph = graph


async def queue(graph, extra, client_id=None):
    if _PREVIEW.get():
        raise _Preview(graph)
    return await comfy_link.queue_prompt(graph, extra, client_id)


SENT_LABELS = {"MiniMaxH3ReferenceToVideo": "Prompt", "MiniMaxH3TextToVideo": "Prompt", "TextGenerateV2": "To the prompt upsampler",
               "MiniMaxH3ImageConditioning": "Prompt", "PH3ImageConditioning": "Prompt"}
PICTURE_LOADERS = {"MMVisionLoadStill": "file", "LoadImage": "image", "MMVisionLoadImage": "file"}


def sent_of(graph):
    """The texts a graph gives its model (prompt, negative, upsampler request) and its pictures, in order."""
    negatives = set()
    for node in graph.values():
        link = node["inputs"].get("negative")
        if isinstance(link, list):
            negatives.add(str(link[0]))
            src = graph.get(str(link[0]), {})
            if src.get("class_type") == "ConditioningZeroOut":
                negatives.add(str(src["inputs"]["conditioning"][0]) + "#zero")
    texts, pictures, mods, unets = [], [], [], []
    for nid in sorted(graph, key=lambda k: (len(k), k)):
        node = graph[nid]
        cls = node["class_type"]
        for key in ("prompt", "text"):
            value = node["inputs"].get(key)
            if isinstance(value, str) and cls not in ("PreviewAny", "SaveImage", "SaveVideo"):
                label = "Negative prompt" if nid in negatives else SENT_LABELS.get(cls, "Prompt")
                texts.append({"label": label, "text": value})
        if cls in PICTURE_LOADERS and isinstance(node["inputs"].get(PICTURE_LOADERS[cls]), str):
            pictures.append(os.path.basename(node["inputs"][PICTURE_LOADERS[cls]]))
        if isinstance(node["inputs"].get("unet_name"), str):
            unets.append(os.path.basename(node["inputs"]["unet_name"]))
        if cls == refmod_bridge.LOADER:
            mods += ["%s (strength %g)" % (node["inputs"]["mod_%d" % n], node["inputs"]["strength_%d" % n])
                     for n in range(1, refmod_bridge.SLOTS + 1) if node["inputs"].get("mod_%d" % n) not in (None, "(none)")]
    return texts, pictures, mods, unets


async def preview(request):
    """POST: the body Generate would send. Answers what the model would get, without rendering anything."""
    token = _PREVIEW.set(True)
    try:
        await generate(request)
    except _Preview as stop:
        texts, pictures, mods, unets = sent_of(stop.graph)
        body = await request.json()
        async with _lock:
            names = {i["file"]: i["name"] for i in read_project()["images"]}
        pictures = ["%s (%s)" % (names[f], f) if f in names else f for f in pictures]
        return web.json_response({"texts": texts, "pictures": pictures, "refmods": mods, "files": unets,
                                  "model": (body.get("settings") or {}).get("model")})
    finally:
        _PREVIEW.reset(token)
    raise web.HTTPBadRequest(text="Nothing would be sent.")


async def generate(request):
    body = await request.json()
    prompt = str(body.get("prompt", "")).strip()
    s = body.get("settings") or {}
    if body.get("kind") == "pruna":
        return await generate_pruna(prompt, s, body)
    if body.get("kind") == "klein" and body.get("mode") == "swap":
        return await generate_klein(prompt, s, body.get("slots") or [], body.get("negative"), swap=True, mask=body.get("mask"))
    if not prompt:
        raise web.HTTPBadRequest(text="Write a prompt first.")
    if body.get("kind") == "image":
        return await generate_image(prompt, s, body.get("negative"), body.get("edit"))
    if body.get("kind") == "klein":
        return await generate_klein(prompt, s, body.get("slots") or [], body.get("negative"), swap=body.get("mode") == "swap")
    models, why = video_models()
    model = next((m for m in models if m["name"] == s.get("model")), None)
    if not model:
        raise web.HTTPBadRequest(text=why or "That model isn't installed.")
    aspect = s.get("aspect") if s.get("aspect") in ASPECTS else "16:9"
    res = round(float(s.get("res", 0.6)), 2)
    if not VIDEO_MP["min"] <= res <= VIDEO_MP["max"]:
        raise web.HTTPBadRequest(text="The size must be %g to %g MP." % (VIDEO_MP["min"], VIDEO_MP["max"]))
    steps = int(s.get("steps", model["steps"]))
    low, high, every = model.get("minSteps", 1), model["maxSteps"], model.get("stepSize", 1)
    if not (low <= steps <= high and (steps - low) % every == 0):
        raise web.HTTPBadRequest(text="%s takes %s steps." % (model["name"], ", ".join(map(str, range(low, high + 1, every)))
                                                                if every > 1 else "%d to %d" % (low, high)))
    sampler, scheduler = pick_sampling(s, model, True)
    length = int(s.get("len", 6))
    if not LENGTH["min"] <= length <= LENGTH["max"]:
        raise web.HTTPBadRequest(text="Length must be %d to %d seconds." % (LENGTH["min"], LENGTH["max"]))
    seed = int(s.get("seed", 0)) if s.get("seedMode") == "fixed" else random.randrange(2 ** 31)
    loras = [{"name": str(l["name"]), "strength": float(l["strength"])} for l in s.get("loras", []) if l.get("on", True)]

    async with _lock:
        data = read_project()
        names = {i["name"] for i in data["images"]}
        pictures = None
        if body.get("pictures") is not None:
            try:
                pictures = h3_graph.clean_pictures(body["pictures"], data["images"], prompt)
            except ValueError as error:
                raise web.HTTPBadRequest(text=str(error))
        else:
            unknown = [n for n in MENTION_RE.findall(prompt) if n not in names]
            if unknown:
                raise web.HTTPBadRequest(text="There is no image called @%s." % unknown[0])
        try:
            refmods = refmod_bridge.clean(s.get("refmods"))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error))
        w, h = video_dims(aspect, res)
        clip = {"id": "c" + uuid.uuid4().hex[:10], "prompt": prompt, "model": model["name"], "profile": model["id"],
                "steps": steps, "res": res, "aspect": aspect, "len": length, "width": w, "height": h, "seed": seed,
                "loras": loras, "status": "queued", "step": 0, "file": None, "subfolder": ROOT + "/clips",
                "starred": False, "created": time.time(), "kind": "video"}
        if sampler:
            clip["sampler"] = sampler
        if scheduler:
            clip["scheduler"] = scheduler
        images = list(data["images"])
        if pictures is not None:
            clip["pictures"] = pictures
        if refmods:
            clip["refmods"] = refmods
        realism = s.get("realism") or {}
        if realism.get("on") and h3_graph.realism_lora():
            low, high = h3_graph.REALISM["range"]
            clip["realism"] = round(max(low, min(high, float(realism.get("strength", h3_graph.REALISM["strength"])))), 2)
        # The pictures it uses, so a project export finds them even after a rename.
        clip["refs"] = ([dict(p) for p in pictures] if pictures is not None
                        else [{"id": i["id"], "name": i["name"]} for i in edit_project.refs_of(clip, images)])
        motion = None
        if body.get("motion"):
            # Motion from a video: Motion Transfer v1.0, the video giving the motion, timing, camera and sound.
            motion = next((c for c in data["clips"] if c["id"] == body["motion"]), None)
            if not motion or motion.get("kind", "video") != "video" or motion.get("status") != "done" or not motion.get("file"):
                raise web.HTTPBadRequest(text="The video to take the motion from isn't there any more.")
            if not pictures:
                raise web.HTTPBadRequest(text="Motion needs at least one picture: <Picture 1> is usually who performs it.")
            if not any(m["id"] == model["id"] for m in swap.models(models)):
                raise web.HTTPBadRequest(text="Motion can't run on HyperFlow, whose schedule is its own: pick another model.")
            method = body.get("motionMethod") if body.get("motionMethod") in swap.METHODS else "guide"
            st = swap.status()
            why = st["modes"]["motion"] if method == "guide" else st["methods"]["tara"]
            if why:
                raise web.HTTPBadRequest(text=why)
            path = clip_path(motion)
            if not os.path.isfile(path):
                raise web.HTTPBadRequest(text="The video file to take the motion from is gone.")
            w, h, frames = swap.guide_shape(path, res)
            if frames < (swap.TARA_MIN_FRAMES if method == "tara" else 5):
                raise web.HTTPBadRequest(text="That video is too short: TARA needs at least 2 seconds." if method == "tara"
                                         else "That video is too short for H3.")
            clip.update({"width": w, "height": h, "len": round(frames / 24, 2), "aspect": motion.get("aspect") or aspect,
                         "prompt": swap.motion_prompt(prompt), "motion": {"from": motion["id"], "method": method}})
            clip.pop("refmods", None)
            clip.pop("realism", None)

    try:
        if motion:
            graph = swap.build(clip, images, "%s/%s" % (motion.get("subfolder") or ROOT + "/clips", motion["file"]), "motion",
                               megapixels=res, name=clip_info.file_stem(clip) + "_motion", model=model["id"],
                               method=clip["motion"]["method"])
            clip["steps"] = graph["12"]["inputs"]["steps"]
        else:
            graph = h3_graph.build_graph(clip, images, name=clip_info.file_stem(clip))
        clip["prompt_id"] = await queue(graph, clip_info.extra_data(clip), body.get("client_id"))
    except h3_graph.Missing as error:
        raise web.HTTPBadRequest(text=str(error))
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't queue a render")
        raise web.HTTPBadRequest(text="ComfyUI refused the render: %s" % error)

    async with _lock:
        data = read_project()
        data["clips"].insert(0, clip)
        write_project(data)
    ensure_watcher()
    broadcast(clip)
    return web.json_response({"clip": clip})


async def generate_image(prompt, s, negative=None, edit=None):
    """A MOSS-Image picture; with edit (an Add image library id), an edit of that picture with the edit LoRA."""
    models, why = image_models()
    model = next((m for m in models if m["name"] == s.get("model")), None)
    if not model:
        raise web.HTTPBadRequest(text=why or "That model isn't installed.")
    aspect = s.get("aspect") if s.get("aspect") in IMAGE_ASPECTS else "1:1"
    res = float(s.get("res", 1.0))
    if res not in IMAGE_MEGAPIXELS:
        raise web.HTTPBadRequest(text="Unknown size.")
    source = None
    if edit:
        # Edit: the picture from the Add image library, and the edit LoRA's settings.
        async with _lock:
            source = next((i for i in read_project()["images"] if i["id"] == edit), None)
        if not source or not os.path.exists(image_path(source)):
            raise web.HTTPBadRequest(text="The picture to edit isn't in the Add image library any more.")
        if not moss_bridge.edit_lora():
            raise web.HTTPBadRequest(text="Edit needs the MOSS-Image edit LoRA (%s) in models/loras "
                                          "(on RunPod: ENABLE_MOSS_EDIT=true)." % moss_bridge.EDIT_LORA)
        model = dict(model, minSteps=moss_bridge.EDIT["minSteps"], maxSteps=moss_bridge.EDIT["maxSteps"])
    steps = int(s.get("steps", model["steps"]))
    if not model["minSteps"] <= steps <= model["maxSteps"]:
        raise web.HTTPBadRequest(text="%s takes %d to %d steps." % (model["name"], model["minSteps"], model["maxSteps"]))
    cfg = float(s.get("cfg", model["cfg"]))
    if not CFG["min"] <= cfg <= CFG["max"]:
        raise web.HTTPBadRequest(text="Guidance must be %s to %s." % (CFG["min"], CFG["max"]))
    seed = int(s.get("seed", 0)) if s.get("seedMode") == "fixed" else random.randrange(2 ** 31)
    loras = [{"name": str(l["name"]), "strength": float(l["strength"])} for l in s.get("loras", []) if l.get("on", True)]
    sampler, scheduler = pick_sampling(s, model, False)
    # The negative prompt is exactly what the box holds; empty means none.
    negative = str(negative).strip()[:2000] if negative is not None else None
    w, h = dims(aspect, res, 16, IMAGE_ASPECTS)
    if source:
        size = picture_size(image_path(source))
        if not size:
            raise web.HTTPBadRequest(text="Couldn't read the picture to edit.")
        w, h = moss_bridge.edit_size(size[0], size[1], res)
        aspect = nearest_aspect(w, h, IMAGE_ASPECTS)
    clip = {"id": "s" + uuid.uuid4().hex[:10], "kind": "image", "prompt": prompt, "model": model["name"], "profile": model["id"],
            "steps": steps, "cfg": cfg, "res": res, "aspect": aspect, "width": w, "height": h, "seed": seed, "loras": loras,
            "status": "queued", "step": 0, "file": None, "subfolder": ROOT + "/stills", "starred": False,
            "created": time.time()}
    for key, value in (("sampler", sampler), ("scheduler", scheduler)):
        if value:
            clip[key] = value
    if negative is not None:
        clip["negative"] = negative
    if source:
        clip["edit"] = {"id": source["id"], "name": source["name"]}
    try:
        graph = moss_bridge.build_graph(clip, name=clip_info.file_stem(clip),
                                        source="%s/images/%s" % (ROOT, source["file"]) if source else None)
        clip["prompt_id"] = await queue(graph, clip_info.extra_data(clip, moss_bridge.EXTRA_DATA))
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't queue an image")
        raise web.HTTPBadRequest(text="ComfyUI refused the image: %s" % error)
    async with _lock:
        data = read_project()
        data["clips"].insert(0, clip)
        write_project(data)
    ensure_watcher()
    broadcast(clip)
    return web.json_response({"clip": clip})


def picture_size(path):
    """A picture's width and height, or None."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def size_like(w0, h0, mp, grid=16):
    """Width and height with the shape of a w0 x h0 picture, at about mp megapixels, in steps of grid."""
    pixels = mp * 1024 * 1024
    k = math.sqrt(pixels / (w0 * h0))
    return max(grid, round(w0 * k / grid) * grid), max(grid, round(h0 * k / grid) * grid)


async def generate_klein(prompt, s, slots, negative=None, swap=False, mask=None):
    """A Klein 9B image: an edit of up to four pictures (the slots image 1-4), or from the prompt alone.
    swap: a face swap with the BFS LoRA, picture 1 the body and scene, picture 2 the head; the prompt is notes."""
    models, why = klein_models()
    model = next((m for m in models if m["name"] == s.get("model")), None)
    if not model:
        raise web.HTTPBadRequest(text=why or "That Klein model isn't installed.")
    res = float(s.get("swapRes", 2.0) if swap else s.get("res", 1.0))
    if res not in KLEIN_MEGAPIXELS:
        raise web.HTTPBadRequest(text="Unknown size.")
    steps = int(s.get("steps", model["steps"]))
    if not model["minSteps"] <= steps <= model["maxSteps"]:
        raise web.HTTPBadRequest(text="%s takes %d to %d steps." % (model["name"], model["minSteps"], model["maxSteps"]))
    cfg = float(s.get("cfg", model["cfg"])) if model["base"] else model["cfg"]
    if not CFG["min"] <= cfg <= CFG["max"]:
        raise web.HTTPBadRequest(text="Guidance must be %s to %s." % (CFG["min"], CFG["max"]))
    size = s.get("size") if s.get("size") == "match" or s.get("size") in IMAGE_ASPECTS else "match"
    seed = int(s.get("seed", 0)) if s.get("seedMode") == "fixed" else random.randrange(2 ** 31)
    loras = [{"name": str(l["name"]), "strength": float(l["strength"])} for l in s.get("loras", []) if l.get("on", True)]
    count = klein_bridge.SWAP_SLOTS if swap else klein_bridge.SLOTS
    slots = (list(slots) + [None] * count)[:count]
    if swap:
        if not klein_bridge.bfs_lora():
            raise web.HTTPBadRequest(text="Face swap needs the BFS head-swap LoRA for Klein 9B "
                                          "(bfs_head_v1_flux-klein_9b_step3750_rank64.safetensors) in models/loras.")
        if not (slots[0] and slots[1]):
            raise web.HTTPBadRequest(text="Face swap needs both pictures: the body and scene, and the face.")

    async with _lock:
        data = read_project()
        by_id = {i["id"]: i for i in data["images"]}
    pictures = []
    for n, image_id in enumerate(slots, 1):
        if not image_id:
            continue
        image = by_id.get(image_id)
        if not image:
            raise web.HTTPBadRequest(text="The picture in image %d isn't in the Add image library any more." % n)
        if not os.path.exists(image_path(image)):
            raise web.HTTPBadRequest(text="The picture in image %d (%s) is missing its file." % (n, image["name"]))
        pictures.append((n, image))
    empty = [] if swap else sorted({int(m) for m in re.findall(r"\bimage\s?([1-4])\b", prompt, re.I)} - {n for n, _ in pictures})
    if empty:
        raise web.HTTPBadRequest(text="The prompt mentions %s, but %s empty." % (
            " and ".join("image %d" % n for n in empty), "those slots are" if len(empty) > 1 else "that slot is"))

    # Size: image 1's shape (as the Klein templates do), or an aspect ratio. With no image 1, square.
    first = picture_size(image_path(pictures[0][1])) if pictures and pictures[0][0] == 1 else None
    if size == "match" and first:
        w, h = size_like(first[0], first[1], res)
        aspect = nearest_aspect(w, h, IMAGE_ASPECTS)
    else:
        aspect = size if size != "match" else "1:1"
        w, h = dims(aspect, res, 16, IMAGE_ASPECTS)
    clip = {"id": "k" + uuid.uuid4().hex[:10], "kind": "image", "family": "klein", "prompt": prompt or ("Face swap" if swap else ""), "model": model["name"],
            "profile": model["id"], "steps": steps, "cfg": cfg, "res": res, "size": size, "aspect": aspect, "width": w,
            "height": h, "seed": seed, "loras": loras, "status": "queued", "step": 0, "file": None,
            "subfolder": ROOT + "/stills", "starred": False, "created": time.time(),
            "slots": [next(({"id": i["id"], "name": i["name"]} for k, i in pictures if k == n), None)
                      for n in range(1, count + 1)]}
    if swap:
        # Face swap: which BFS LoRA, the head mask (only the head changes) with LanPaint or the plain sampler,
        # and Refine face. The pictures 3-5 are more of the same face.
        clip.update({"swap": True, "sent": klein_bridge.swap_prompt(prompt, len(pictures)),
                     "bfs": klein_bridge.bfs_lora(s.get("swapLora")), "lanSteps": 20})
        if mask:
            file = "%s/masks/%s" % (ROOT, os.path.basename(str(mask)))
            if not os.path.exists(os.path.join(output_dir(), file)):
                raise web.HTTPBadRequest(text="The head mask is missing: paint it again.")
            clip["mask"] = {"file": file, "grow": int(s.get("maskGrow", 8)), "feather": int(s.get("maskFeather", 12))}
            if s.get("swapSampler") == "lanpaint":
                if not klein_bridge.lanpaint():
                    raise web.HTTPBadRequest(text="LanPaint isn't installed (scraed/LanPaint in custom_nodes).")
                clip["sampler"] = "lanpaint"
                clip["lanSteps"] = max(4, min(60, int(s.get("lanSteps", 20))))
                clip["steps"] = clip["lanSteps"]
            if s.get("refine"):
                clip["refine"] = True
                clip["refineStrength"] = max(0.2, min(0.9, float(s.get("refineStrength", 0.5))))
    if model["base"] and negative is not None:
        clip["negative"] = str(negative).strip()[:2000]
    refs = [(n, "%s/images/%s" % (ROOT, image["file"])) for n, image in pictures]
    try:
        graph = klein_bridge.build_graph(clip, refs, name=clip_info.file_stem(clip))
        clip["prompt_id"] = await queue(graph, clip_info.extra_data(clip, klein_bridge.EXTRA_DATA))
    except klein_bridge.Missing as error:
        raise web.HTTPBadRequest(text=str(error))
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't queue a Klein image")
        raise web.HTTPBadRequest(text="ComfyUI refused the image: %s" % error)
    async with _lock:
        data = read_project()
        data["clips"].insert(0, clip)
        write_project(data)
    ensure_watcher()
    broadcast(clip)
    return web.json_response({"clip": clip})


async def generate_pruna(prompt, s, body):
    """A P-H3-Image picture: from the prompt, an edit of picture 1 with up to eight more as references, or a
    character sheet (the character and its helper pictures, with the sheet's text written by the app)."""
    models, why = pruna_models()
    model = next((m for m in models if m["name"] == s.get("model")), None) or (models[0] if models else None)
    if not model:
        raise web.HTTPBadRequest(text=why or "P-H3-Image isn't installed.")
    sheet = body.get("mode") == "sheet"
    res = float(s.get("sheetRes" if sheet else "res", 8.0 if sheet else 4.0))
    if res not in model.get("megapixels", ph3_bridge.MEGAPIXELS):
        raise web.HTTPBadRequest(text="%s doesn't come in %g MP." % (model["name"], res))
    runs = [model]
    enhance = bool(s.get("enhance")) and not model.get("h3image")   # H3-Image reads the prompt as written
    if enhance and not ph3_bridge.upsampler():
        raise web.HTTPBadRequest(text="Enhance prompt needs Pruna's prompt upsampler "
                                      "(p-h3-image_prompt_upsampler_bf16.safetensors in models/text_encoders).")
    size = s.get("size") if s.get("size") == "match" or s.get("size") in PH3_ASPECTS else "match"
    seed = int(s.get("seed", 0)) if s.get("seedMode") == "fixed" else random.randrange(2 ** 31)
    loras = [{"name": str(l["name"]), "strength": float(l["strength"])} for l in s.get("loras", []) if l.get("on", True)]
    count = 4 if sheet else ph3_bridge.SLOTS
    slots = (list(body.get("slots") or []) + [None] * count)[:count]
    sheet_in = body.get("sheet") or {}
    if sheet:
        text = str(sheet_in.get("text") or "").strip()
        if not text:
            raise web.HTTPBadRequest(text="Pick at least one view.")
        if not slots[0]:
            raise web.HTTPBadRequest(text="Add the character picture first.")
    elif not prompt:
        raise web.HTTPBadRequest(text="Write a prompt first.")

    async with _lock:
        data = read_project()
        by_id = {i["id"]: i for i in data["images"]}
    pictures = []
    for n, image_id in enumerate(slots, 1):
        if not image_id:
            continue
        image = by_id.get(image_id)
        if not image:
            raise web.HTTPBadRequest(text="The picture in slot %d isn't in the Add image library any more." % n)
        if not os.path.exists(image_path(image)):
            raise web.HTTPBadRequest(text="The picture in slot %d (%s) is missing its file." % (n, image["name"]))
        pictures.append((n, image))
    if not sheet:
        empty = sorted({int(m) for m in ph3_bridge.PICTURE_RE.findall(prompt)} - {n for n, _ in pictures})
        if empty:
            raise web.HTTPBadRequest(text="The prompt mentions %s, but %s empty." % (
                " and ".join("picture %d" % n for n in empty), "those slots are" if len(empty) > 1 else "that slot is"))

    if sheet:
        ratio = float(sheet_in.get("ratio") or 2.5)
        if not 0.3 <= ratio <= 6:
            raise web.HTTPBadRequest(text="That sheet shape is too narrow or too wide.")
    first = picture_size(image_path(pictures[0][1])) if pictures and pictures[0][0] == 1 else None

    def size_at(mp):
        if sheet:
            w, h = ph3_bridge.size_for(ratio * 1000, 1000, mp)
            return w, h, ph3_bridge.ratio_name(w, h)
        if size == "match" and first:
            w, h = ph3_bridge.size_for(first[0], first[1], mp)
            return w, h, nearest_aspect(w, h, PH3_ASPECTS)
        aspect = size if size != "match" else "1:1"
        w, h = dims(aspect, mp, ph3_bridge.GRID, PH3_ASPECTS)
        return w, h, aspect

    text = sheet_in["text"].strip()[:6000] if sheet else ph3_bridge.model_prompt(prompt, [n for n, _ in pictures])
    refs = ["%s/images/%s" % (ROOT, image["file"]) for _, image in pictures]
    made = []
    for run in runs:
        mp = res
        w, h, aspect = size_at(mp)
        h3image = bool(run.get("h3image"))
        clip = {"id": "p" + uuid.uuid4().hex[:10], "kind": "image", "family": "ph3", "mode": "sheet" if sheet else "image",
                "prompt": prompt or "Character sheet", "model": run["name"], "profile": run["id"],
                "steps": run["steps"], "cfg": 1.0, "res": mp, "size": "sheet" if sheet else size, "aspect": aspect,
                "width": w, "height": h, "seed": seed, "loras": loras if run.get("loras", True) else [],
                "enhance": enhance and not h3image,
                "sent": ph3_bridge.sent_text(text, bool(pictures), enhance and not h3image, w, h, h3image),
                "status": "queued", "step": 0, "file": None, "subfolder": ROOT + "/stills", "starred": False,
                "created": time.time() + 0.001 * len(made),
                "slots": [next(({"id": i["id"], "name": i["name"]} for k, i in pictures if k == n), None) for n in range(1, count + 1)]}
        if sheet:
            roles = (list(sheet_in.get("roles") or []) + ["character", "face", "outfit", "detail"])[:4]
            clip["sheet"] = {"views": [str(v) for v in sheet_in.get("views") or []][:16], "preset": str(sheet_in.get("preset") or ""),
                             "bg": str(sheet_in.get("bg") or "white"), "roles": [str(r) for r in roles], "ratio": ratio}
        try:
            graph = ph3_bridge.build_graph(clip, refs, name=clip_info.file_stem(clip))
            clip["prompt_id"] = await queue(graph, clip_info.extra_data(clip, ph3_bridge.EXTRA_DATA))
        except ph3_bridge.Missing as error:
            raise web.HTTPBadRequest(text=str(error))
        except Exception as error:
            LOG.exception("MiniMax H3 Vision couldn't queue a P-H3-Image picture")
            raise web.HTTPBadRequest(text="ComfyUI refused the image: %s" % error)
        made.append(clip)
    async with _lock:
        data = read_project()
        for clip in made:
            data["clips"].insert(0, clip)
        write_project(data)
    ensure_watcher()
    for clip in made:
        broadcast(clip)
    return web.json_response({"clip": made[0], "clips": made})


def name_from(prompt, taken):
    """An @name for a picture, from its quoted words (or its first words), cut at a whole word."""
    quoted = re.search(r'["\u201c]([^"\u201d]{1,60})["\u201d]', prompt)
    words = re.findall(r"[a-z0-9]+", (quoted.group(1) if quoted else prompt).lower())
    base = ""
    for word in words:
        if base and len(base) + 1 + len(word) > 20:
            break
        base = (base + "_" + word) if base else word[:20]
    base = base or "image"
    name, n = base, 2
    while name in taken:
        name, n = "%s%d" % (base, n), n + 1
    return name


async def image_from_clip(request):
    """A finished image from the Image tab, as an @image for videos. Made once per image; asked again, it is returned."""
    clip_id = str((await request.json()).get("clip_id", ""))
    async with _lock:
        data = read_project()
        had = next((i for i in data["images"] if i.get("from_clip") == clip_id), None)
        if had:
            return web.json_response({"image": had})
        clip = next((c for c in data["clips"] if c["id"] == clip_id), None)
        if not clip or clip.get("kind") != "image" or clip.get("status") != "done" or not clip.get("file"):
            raise web.HTTPBadRequest(text="That image isn't finished.")
        source = os.path.join(output_dir(), clip.get("subfolder") or ROOT + "/stills", clip["file"])
        image_id = "i" + uuid.uuid4().hex[:10]
        file = image_id + (os.path.splitext(clip["file"])[1] or ".png")
        shutil.copyfile(source, root_dir("images", file))
        image = {"id": image_id, "name": name_from(clip["prompt"], {i["name"] for i in data["images"]}), "file": file,
                 "added": time.time(), "from_clip": clip_id}
        data["images"].append(image)
        write_project(data)
    return web.json_response({"image": image})


def nearest(value, options):
    return min(options, key=lambda o: abs(o - value))


def nearest_aspect(w, h, table):
    return min(table, key=lambda k: abs(math.log((w / h) / (table[k][0] / table[k][1]))))


async def import_clip(request):
    """A video or image made earlier (here, in the H3 editor or by MiniMax H3 Vision elsewhere): it joins the
    grid with the prompt and settings found inside it."""
    reader = await request.multipart()
    part = await reader.next()
    while part is not None and not part.filename:
        part = await reader.next()
    if part is None:
        raise web.HTTPBadRequest(text="No file came with the import.")
    ext = os.path.splitext(part.filename)[1].lower()
    kind = "video" if ext in clip_info.VIDEO_TYPES else "image" if ext in clip_info.IMAGE_TYPES else None
    if not kind:
        raise web.HTTPBadRequest(text="Import takes a video (.mp4, .mov, .webm) or an image (.png, .jpg, .webp).")
    folder = "clips" if kind == "video" else "stills"
    temp = root_dir(folder, ".import_%s%s" % (uuid.uuid4().hex[:8], ext))
    with open(temp, "wb") as f:
        while True:
            chunk = await part.read_chunk()
            if not chunk:
                break
            f.write(chunk)
    try:
        found = clip_info.settings(temp)
    except Exception:
        LOG.exception("MiniMax H3 Vision couldn't read an imported file")
        found = None
    plain = None
    if (not found or not found.get("prompt")) and kind == "video":
        # Any other video comes in as it is, without a prompt: to watch, edit, or swap a character into.
        try:
            plain, temp, ext = await asyncio.to_thread(plain_video, temp)
        except Exception as error:
            LOG.exception("MiniMax H3 Vision couldn't read an imported video")
            if os.path.exists(temp):
                os.remove(temp)
            raise web.HTTPBadRequest(text="%s couldn't be read as a video (%s)." % (part.filename, error))
        found = {"prompt": "", "width": plain["width"], "height": plain["height"], "len": plain["seconds"],
                 "model": "Imported video", "from": "video file"}
    elif not found or not found.get("prompt"):
        os.remove(temp)
        raise web.HTTPBadRequest(text="%s has no prompt or settings inside it. Only images saved by ComfyUI with its "
                                      "metadata (from MiniMax H3 Vision or a MOSS-Image workflow) have them. To use a "
                                      "picture in a video, add it with Add image." % part.filename)
    if (found.get("kind") or kind) != kind:
        found["kind"] = kind
    clip = clip_from_settings(found, kind)
    stem = clip_info.file_stem(clip)
    if plain:
        clip.update({"len": plain["seconds"], "steps": 0})
        stem = "import_" + (clip_info.slug(os.path.splitext(part.filename)[0]) or "video")
    file = numbered(root_dir(folder), stem, ext)
    os.replace(temp, root_dir(folder, file))
    clip.update({"id": ("c" if kind == "video" else "k" if clip.get("family") == "klein" else "p" if clip.get("family") == "ph3" else "s") + uuid.uuid4().hex[:10],
                 "status": "done", "step": clip["steps"],
                 "file": file, "subfolder": ROOT + "/" + folder, "starred": False, "created": time.time(),
                 "mode": "imported", "imported": part.filename, "found": found["from"]})
    async with _lock:
        data = read_project()
        data["clips"].insert(0, clip)
        write_project(data)
    broadcast(clip)
    return web.json_response({"clip": clip})


PLAYABLE = {("mp4", "h264"), ("mp4", "av1"), ("mov", "h264"), ("webm", "vp8"), ("webm", "vp9"), ("webm", "av1")}


def plain_video(path):
    """A video file without Vision's settings: its size and length, re-encoded as H.264 .mp4 when a browser
    couldn't play it as it is. Returns (facts, path, extension)."""
    import av
    ext = os.path.splitext(path)[1].lower()
    with av.open(path) as container:
        stream = container.streams.video[0]
        width, height = stream.codec_context.width, stream.codec_context.height
        seconds = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / 1e6
        codec = stream.codec_context.name
    if (ext.lstrip(".").replace("m4v", "mp4"), codec) not in PLAYABLE:
        out = os.path.splitext(path)[0] + "_h264.mp4"
        to_h264(path, out)
        os.remove(path)
        path, ext = out, ".mp4"
    return {"width": width, "height": height, "seconds": round(seconds, 2)}, path, ext


def to_h264(source, out):
    """Re-encode a video as H.264 (CRF 18) with AAC sound in an .mp4, so a browser plays it."""
    import av
    with av.open(source) as src, av.open(out, "w") as dst:
        video = src.streams.video[0]
        audio = src.streams.audio[0] if src.streams.audio else None
        rate = video.average_rate or video.guessed_rate or 24
        v_out = dst.add_stream("libx264", rate=rate, options={"crf": "18", "preset": "veryfast"})
        v_out.width, v_out.height = video.codec_context.width // 2 * 2, video.codec_context.height // 2 * 2
        v_out.pix_fmt = "yuv420p"
        a_out = dst.add_stream("aac", rate=audio.codec_context.sample_rate or 48000) if audio else None
        step, count = 1 / fractions.Fraction(rate), 0
        for packet in src.demux([s for s in (video, audio) if s]):
            for frame in packet.decode():
                if packet.stream is video:
                    frame = frame.reformat(width=v_out.width, height=v_out.height, format="yuv420p")
                    frame.pts, frame.time_base, count = count, step, count + 1
                    dst.mux(v_out.encode(frame))
                else:
                    frame.pts = None
                    dst.mux(a_out.encode(frame))
        dst.mux(v_out.encode())
        if a_out:
            dst.mux(a_out.encode())


def clip_from_settings(found, kind):
    """A clip record from what a file says, filled in to match this app's settings."""
    klein = kind == "image" and found.get("family") == "klein"
    ph3 = kind == "image" and found.get("family") == "ph3"
    if ph3:
        models, _ = pruna_models()
        table, sizes, default_aspect = PH3_ASPECTS, sorted(set(ph3_bridge.MEGAPIXELS + ph3_bridge.H3IMAGE_MEGAPIXELS)), "1:1"
    elif klein:
        models, _ = klein_models()
        table, sizes, default_aspect = IMAGE_ASPECTS, KLEIN_MEGAPIXELS, "1:1"
    elif kind == "image":
        models, _ = image_models()
        table, sizes, default_aspect = IMAGE_ASPECTS, IMAGE_MEGAPIXELS, "1:1"
    else:
        models, _ = video_models()
        table, sizes, default_aspect = ASPECTS, MEGAPIXELS, "16:9"
    model = next((m for m in models if m["id"] == found.get("profile") or m["name"] == found.get("model")), None)
    if not model and found.get("unet"):
        # An H3 editor graph names a checkpoint file: find the model family that uses it.
        try:
            model = next((m for m in models if m["id"] == h3_graph.profile_for(found["unet"], found.get("unet_lora"))), None)
        except Exception:
            model = None
    w, h = found.get("width") or 0, found.get("height") or 0
    aspect = found.get("aspect") if found.get("aspect") in table else nearest_aspect(w, h, table) if w and h else default_aspect
    res = found.get("res") if found.get("res") in sizes else nearest((w * h) / (1024 * 1024), sizes) if w and h else sizes[0]
    if kind != "image" and isinstance(found.get("res"), (int, float)) and VIDEO_MP["min"] <= found["res"] <= VIDEO_MP["max"]:
        res = round(float(found["res"]), 2)   # a typed video size, like 0.98
    clip = {"kind": kind, "prompt": found["prompt"],
            "model": model["name"] if model else found.get("model") or os.path.splitext(os.path.basename(found.get("profile") or found.get("unet") or "Unknown model"))[0],
            "profile": model["id"] if model else found.get("profile"),
            "steps": int(found.get("steps") or (model or {}).get("steps") or 0), "res": res, "aspect": aspect,
            "width": w or None, "height": h or None, "seed": int(found.get("seed") or 0),
            "loras": [{"name": str(l["name"]), "strength": float(l.get("strength", 1.0))} for l in found.get("loras") or []]}
    if kind == "image":
        clip["cfg"] = float(found.get("cfg") or (model or {}).get("cfg") or 3.0)
        if found.get("edit"):
            clip["edit"] = found["edit"]
    if klein:
        # Its pictures are named; they are found again by id when this machine has them.
        clip.update({"family": "klein", "size": found.get("size") or "match",
                     "slots": (list(found.get("slots") or []) + [None] * klein_bridge.SLOTS)[:klein_bridge.SLOTS]})
    if klein and found.get("swap"):
        clip.update({"swap": True, "sent": found.get("sent") or "",
                     "slots": (list(found.get("slots") or []) + [None] * klein_bridge.SWAP_SLOTS)[:klein_bridge.SWAP_SLOTS]})
        for key in ("bfs", "mask", "sampler", "lanSteps", "refine", "refineStrength"):
            if key in found:
                clip[key] = found[key]
    if ph3:
        # Its pictures are named, found again by id when this machine has them; a sheet keeps its views.
        count = 4 if found.get("mode") == "sheet" else ph3_bridge.SLOTS
        clip.update({"family": "ph3", "mode": found.get("mode") or "image", "size": found.get("size") or "match",
                     "enhance": bool(found.get("enhance")), "sent": found.get("sent") or "",
                     "slots": (list(found.get("slots") or []) + [None] * count)[:count]})
        if found.get("sheet"):
            clip["sheet"] = found["sheet"]
    if kind != "image":
        clip["len"] = int(min(LENGTH["max"], max(LENGTH["min"], found.get("len") or 6)))
    return clip


async def update_clip(request):
    clip_id = request.match_info["id"]
    body = await request.json()
    async with _lock:
        data = read_project()
        clip = next((c for c in data["clips"] if c["id"] == clip_id), None)
        if not clip:
            raise web.HTTPNotFound(text="No such clip.")
        if "starred" in body:
            clip["starred"] = bool(body["starred"])
        write_project(data)
    return web.json_response({"clip": clip})


async def delete_clip(request):
    """Delete a clip. A clip that is still queued or rendering is cancelled first."""
    clip_id = request.match_info["id"]
    async with _lock:
        data = read_project()
        clip = next((c for c in data["clips"] if c["id"] == clip_id), None)
        if not clip:
            raise web.HTTPNotFound(text="No such clip.")
        busy = clip.get("status") in ("queued", "rendering")
        data["clips"].remove(clip)
        write_project(data)
    if busy and clip.get("prompt_id"):
        try:
            await comfy_link.cancel(clip["prompt_id"])
        except Exception:
            LOG.exception("MiniMax H3 Vision couldn't stop a render")
    if clip.get("file"):
        try:
            os.remove(os.path.join(output_dir(), clip.get("subfolder") or ROOT + ("/stills" if clip.get("kind") == "image" else "/clips"), clip["file"]))
        except OSError:
            pass
    return web.json_response({"ok": True})


# ---------------------------------------------------------------- rendering

async def set_clip(clip_id, **fields):
    async with _lock:
        data = read_project()
        clip = next((c for c in data["clips"] if c["id"] == clip_id), None)
        if not clip:
            return None
        clip.update(fields)
        write_project(data)
    broadcast(clip)
    return clip


_watcher = None


def ensure_watcher():
    """One watcher for real renders: it follows each in ComfyUI's queue until its video is saved."""
    global _watcher
    if _watcher is None or _watcher.done():
        _watcher = asyncio.get_running_loop().create_task(watch_real())


async def watch_real():
    while True:
        await asyncio.sleep(1.0)
        try:
            async with _lock:
                busy = [c for c in read_project()["clips"] if c.get("status") in ("queued", "rendering") and c.get("prompt_id")]
            if not busy:
                return
            running, queued = comfy_link.queue_state()
            for clip in busy:
                pid = clip["prompt_id"]
                if pid in running:
                    if clip["status"] != "rendering":
                        await set_clip(clip["id"], status="rendering")
                elif pid in queued:
                    continue
                else:
                    if clip.get("kind") == "image":
                        entry = comfy_link.history(pid)
                        file, why = moss_bridge.saved_file(entry), comfy_link.failure(entry)
                    else:
                        entry = comfy_link.history(pid)
                        file, why = video_file(entry), comfy_link.failure(entry)
                    if file:
                        extra = {}
                        if clip.get("family") == "ph3" and clip.get("enhance"):
                            extra["upsampled"] = ph3_bridge.upsampled_text(entry)
                        # A video swap's swap is a dict; a Klein face swap's is True.
                        if isinstance(clip.get("swap"), dict) and clip["swap"].get("described"):
                            extra["prompt"] = swap.written_prompt(entry) or clip["prompt"]
                        await set_clip(clip["id"], status="done", step=clip["steps"], file=file["filename"],
                                       subfolder=file.get("subfolder") or ROOT + ("/stills" if clip.get("kind") == "image" else "/clips"), **extra)
                    else:
                        await set_clip(clip["id"], status="failed", error=why)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("MiniMax H3 Vision couldn't check its renders")


def video_file(entry):
    """The video a finished job saved, as {filename, subfolder}, or None."""
    for value in ((entry or {}).get("outputs") or {}).get(h3_graph.SAVE_NODE, {}).values():
        for found in value if isinstance(value, list) else []:
            if isinstance(found, dict) and found.get("filename"):
                return found
    return None


async def faces(request):
    """GET /faces: whether Refine faces can run here, and the models it can use.
    POST /clips/{id}/faces {model, strength, face, steps}: refine the faces of a finished video into a new clip."""
    found, _ = video_models()
    usable = face_refine.models(found)
    if request.method == "GET":
        return web.json_response({"why": face_refine.why_not(), "models": usable, "strengths": list(face_refine.STRENGTHS)})
    why = face_refine.why_not()
    if why:
        raise web.HTTPBadRequest(text=why)
    body = await request.json()
    model = next((m for m in usable if m["name"] == body.get("model")), None)
    if not model:
        raise web.HTTPBadRequest(text="Pick a model to refine with (HyperFlow can't).")
    strength = body.get("strength") if body.get("strength") in face_refine.STRENGTHS else "normal"
    steps = int(body.get("steps") or model["steps"])
    if not model.get("minSteps", 1) <= steps <= model["maxSteps"]:
        raise web.HTTPBadRequest(text="%s takes %d to %d steps." % (model["name"], model.get("minSteps", 1), model["maxSteps"]))
    async with _lock:
        data = read_project()
        source = next((c for c in data["clips"] if c["id"] == request.match_info["id"]), None)
        if not source or source.get("kind", "video") != "video" or source.get("status") != "done" or not source.get("file"):
            raise web.HTTPBadRequest(text="Only a finished video can have its faces refined.")
        images = list(data["images"])
        face = next((i for i in images if i["id"] == body.get("face")), None) if body.get("face") else None
        if body.get("face") and not face:
            raise web.HTTPBadRequest(text="That face picture isn't in Add image any more.")
    path = clip_path(source)
    if not os.path.isfile(path):
        raise web.HTTPBadRequest(text="The video file of that clip is gone.")
    clip = {k: v for k, v in source.items() if k not in ("prompt_id", "error", "upsampled", "sampler", "scheduler")}
    if model["id"] == source.get("profile"):
        clip.update({k: source[k] for k in ("sampler", "scheduler") if source.get(k)})
    clip.update({"id": "c" + uuid.uuid4().hex[:10], "model": model["name"], "profile": model["id"], "steps": steps,
                 "status": "queued", "step": 0, "file": None, "subfolder": ROOT + "/clips", "starred": False,
                 "created": time.time(), "faces": {"from": source["id"], "strength": strength,
                                                   "face": face["id"] if face else None}})
    if face and not any(r.get("id") == face["id"] for r in clip.get("refs") or []):
        clip["refs"] = list(clip.get("refs") or []) + [{"id": face["id"], "name": face["name"]}]
    try:
        graph = face_refine.build(clip, images, "%s/%s" % (source.get("subfolder") or ROOT + "/clips", source["file"]),
                                  face_refine.has_audio(path), face="%s/images/%s" % (ROOT, face["file"]) if face else None,
                                  strength=strength, name=clip_info.file_stem(clip) + "_faces")
        clip["prompt_id"] = await queue(graph, clip_info.extra_data(clip), body.get("client_id"))
    except (h3_graph.Missing, ValueError) as error:
        raise web.HTTPBadRequest(text=str(error))
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't queue a face refine")
        raise web.HTTPBadRequest(text="ComfyUI refused the refine: %s" % error)
    async with _lock:
        data = read_project()
        at = next((n for n, c in enumerate(data["clips"]) if c["id"] == source["id"]), 0)
        data["clips"].insert(at, clip)
        write_project(data)
    ensure_watcher()
    broadcast(clip)
    return web.json_response({"clip": clip})


async def swaps(request):
    """GET /swap: which swaps can run here. POST /clips/{id}/swap {mode, picture, details, res}: swap the picture's
    character (or head) into a finished or imported video, as a new clip."""
    found, _ = video_models()
    usable = swap.models(found)
    if request.method == "GET":
        return web.json_response(dict(swap.status(), models=[dict(m, turbo=swap.turbo(m["id"])) for m in usable], base=swap.BASE))
    body = await request.json()
    model = next((m for m in usable if m["id"] == (body.get("model") or swap.BASE)), None)
    if not model:
        raise web.HTTPBadRequest(text="Swap can't run on HyperFlow, whose schedule is its own: pick another model." if usable
                                 else "Swap needs a MiniMax H3 model other than HyperFlow: MiniMax H3 + H3 Turbo is the author's.")
    mode = body.get("mode")
    if mode not in swap.MODES or mode == "motion":
        raise web.HTTPBadRequest(text="Pick Character or Head.")
    why = swap.status()["modes"][mode]
    if why:
        raise web.HTTPBadRequest(text=why)
    res = float(body.get("res") or 0.6)
    if res not in swap.MEGAPIXELS:
        raise web.HTTPBadRequest(text="Pick a resolution from the list.")
    details = str(body.get("details") or "").strip()[:2000]
    mark = clean_mark(body.get("mark")) if mode == "character" else None
    async with _lock:
        data = read_project()
        source = next((c for c in data["clips"] if c["id"] == request.match_info["id"]), None)
        if not source or source.get("kind", "video") != "video" or source.get("status") != "done" or not source.get("file"):
            raise web.HTTPBadRequest(text="Only a finished video can be the guide of a swap.")
        images = list(data["images"])
        by_id = {i["id"]: i for i in images}
        cast = [by_id[body["picture"]]] if body.get("picture") in by_id else []
        if not cast:
            raise web.HTTPBadRequest(text="Pick the picture of the %s." % ("character" if mode == "character" else "head"))
        picture = cast[0]
    path = clip_path(source)
    if not os.path.isfile(path):
        raise web.HTTPBadRequest(text="The video file of that clip is gone.")
    width, height, length = swap.guide_shape(path, res)
    if length < 5:
        raise web.HTTPBadRequest(text="That video is too short for H3.")
    clip = {"id": "c" + uuid.uuid4().hex[:10], "kind": "video", "model": swap.MODES[mode]["name"] + " · " + model["name"],
            "profile": model["id"], "steps": swap.STEPS, "res": res, "aspect": source.get("aspect"), "width": width, "height": height,
            "len": round(length / 24, 2), "seed": random.randint(0, 2**31 - 1), "loras": [],
            "pictures": [{"id": i["id"], "name": i["name"]} for i in cast], "refs": [{"id": i["id"], "name": i["name"]} for i in cast],
            "status": "queued", "step": 0, "file": None, "subfolder": ROOT + "/clips", "starred": False, "created": time.time(),
            "swap": {"from": source["id"], "mode": mode, "picture": picture["id"], "details": details or None, "described": not details,
                     "mark": mark}}
    clip["prompt"] = swap.prompt_of(mode, details, bool(mark)) if details else swap.template_of(mode, bool(mark))
    try:
        graph = swap.build(clip, images, "%s/%s" % (source.get("subfolder") or ROOT + "/clips", source["file"]), mode,
                           details=details, megapixels=res, name=clip_info.file_stem(clip) + "_swap", mark=mark, model=model["id"])
        clip["steps"] = graph["12"]["inputs"]["steps"]
        clip["prompt_id"] = await queue(graph, clip_info.extra_data(clip), None)
    except (h3_graph.Missing, ValueError) as error:
        raise web.HTTPBadRequest(text=str(error))
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't queue a swap")
        raise web.HTTPBadRequest(text="ComfyUI refused the swap: %s" % error)
    async with _lock:
        data = read_project()
        at = next((n for n, c in enumerate(data["clips"]) if c["id"] == source["id"]), 0)
        data["clips"].insert(at, clip)
        write_project(data)
    ensure_watcher()
    broadcast(clip)
    return web.json_response({"clip": clip})


def clean_mark(mark):
    """A point on a video's first frame (0 to 1 across and down), and how solid its colour is; None for no mark."""
    if not isinstance(mark, dict):
        return None
    try:
        x, y = float(mark["x"]), float(mark["y"])
        opacity = float(mark.get("opacity", 1.0))
    except (KeyError, TypeError, ValueError):
        raise web.HTTPBadRequest(text="Click the person on the first frame.")
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise web.HTTPBadRequest(text="Click the person on the first frame.")
    return {"x": round(x, 4), "y": round(y, 4), "opacity": 0.5 if opacity < 1 else 1.0}


async def swap_mark(request):
    """POST /clips/{id}/swap-mark {mark, res}: the video with the person at the point painted red, to check before a
    swap. GET /swap-mark/{prompt_id}: {state: working|done|failed, file, subfolder, error}."""
    if request.method == "GET":
        pid = request.match_info["pid"]
        running, waiting = comfy_link.queue_state()
        if pid in running or pid in waiting:
            return web.json_response({"state": "working"})
        entry = comfy_link.history(pid)
        file = video_file(entry)
        if file:
            return web.json_response({"state": "done", "file": file["filename"], "subfolder": file.get("subfolder") or ""})
        return web.json_response({"state": "failed", "error": comfy_link.failure(entry)})
    body = await request.json()
    mark = clean_mark(body.get("mark"))
    if not mark:
        raise web.HTTPBadRequest(text="Click the person on the first frame.")
    res = float(body.get("res") or 0.6)
    async with _lock:
        source = next((c for c in read_project()["clips"] if c["id"] == request.match_info["id"]), None)
    if not source or not source.get("file"):
        raise web.HTTPBadRequest(text="That video isn't there any more.")
    try:
        graph = swap.mark_graph("%s/%s" % (source.get("subfolder") or ROOT + "/clips", source["file"]), mark, res,
                                "%s/marks/%s" % (ROOT, source["id"]))
        pid = await comfy_link.queue_prompt(graph, {}, None)
    except (h3_graph.Missing, comfy_link.Refused) as error:
        raise web.HTTPBadRequest(text=str(error))
    return web.json_response({"prompt_id": pid})


def numbered(folder, stem, ext):
    """A free file name the way ComfyUI numbers its outputs: stem_00001_.ext, stem_00002_.ext, ..."""
    n = 1
    while os.path.exists(os.path.join(folder, "%s_%05d_%s" % (stem, n, ext))):
        n += 1
    return "%s_%05d_%s" % (stem, n, ext)


# ---------------------------------------------------------------- wiring

# ---------------------------------------------------------------- the Edit tab

async def put_edit(request):
    """Save the edit (the app sends it a moment after each change)."""
    body = await request.json()
    pieces = []
    for p in body.get("pieces") or []:
        try:
            pieces.append({"id": str(p["id"])[:40], "clip": str(p["clip"])[:40], "in": round(float(p["in"]), 4), "out": round(float(p["out"]), 4)})
        except (KeyError, TypeError, ValueError):
            raise web.HTTPBadRequest(text="A piece of the edit is missing its clip or times.")
    edit = {"name": str(body.get("name") or "Untitled edit")[:80], "pieces": pieces[:500],
            "ratio": body.get("ratio") if body.get("ratio") in EDIT_SIZES else "16:9", "muted": bool(body.get("muted"))}
    async with _lock:
        data = read_project()
        data["edit"] = edit
        write_project(data)
    return web.json_response({"edit": edit})


def image_path(image):
    return root_dir("images", image["file"])


def clip_path(clip):
    folder = clip.get("subfolder") or ROOT + "/clips"
    return os.path.join(output_dir(), folder, clip["file"])


async def clip_strip(request):
    """A video's filmstrip and waveform for the timeline (made once, then kept in minimax_flow/strips)."""
    clip_id = request.match_info["id"]
    async with _lock:
        clip = next((c for c in read_project()["clips"] if c["id"] == clip_id), None)
    if not clip or not clip.get("file") or clip.get("kind") == "image":
        raise web.HTTPNotFound(text="No such video.")
    try:
        info = await asyncio.to_thread(edit_media.strip, clip_path(clip), root_dir("strips"), clip_id)
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't read %s", clip.get("file"))
        raise web.HTTPBadRequest(text="Couldn't read %s: %s" % (clip.get("file"), error))
    info = dict(info, sprite_url="/view?type=output&subfolder=%s&filename=%s&v=%s" % (ROOT + "/strips", info["sprite"], int(info["source_mtime"])))
    return web.json_response(info)


# Export sizes per frame shape: "as made" is H3's own 0.6 MP size.
EDIT_SIZES = {"16:9": [[1056, 608], [1280, 720], [1920, 1080]], "9:16": [[608, 1056], [720, 1280], [1080, 1920]],
              "1:1": [[800, 800], [1080, 1080]]}


async def start_export(request):
    """Export the edit as one MP4 in minimax_flow/exports. Progress arrives as mmvision_export events."""
    from server import PromptServer

    body = await request.json()
    width, height = int(body.get("width", 1056)), int(body.get("height", 608))
    if not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise web.HTTPBadRequest(text="Unknown size.")
    async with _lock:
        data = read_project()
    clips = {c["id"]: c for c in data["clips"]}
    pieces = []
    for p in data["edit"]["pieces"]:
        clip = clips.get(p["clip"])
        if clip and clip.get("file") and os.path.exists(clip_path(clip)):
            pieces.append((clip_path(clip), float(p["in"]), float(p["out"])))
    if not pieces:
        raise web.HTTPBadRequest(text="The edit is empty.")
    stem = re.sub(r"[^a-z0-9]+", "_", str(body.get("name") or data["edit"]["name"]).lower()).strip("_")[:60] or "edit"
    folder = root_dir("exports")
    file = stem + ".mp4"
    n = 2
    while os.path.exists(os.path.join(folder, file)):
        file = "%s_%d.mp4" % (stem, n)
        n += 1
    job = "x" + uuid.uuid4().hex[:10]
    _exports[job] = {"cancel": False}
    server, loop = PromptServer.instance, asyncio.get_running_loop()

    def tell(**fields):
        # From the export thread: ComfyUI's send_sync is safe to call from another thread.
        server.send_sync(EXPORT_EVENT, dict(fields, job=job), None)

    async def run():
        try:
            await asyncio.to_thread(edit_media.export, pieces, os.path.join(folder, file), width, height, bool(body.get("sound", True)),
                                    lambda fraction, words: tell(progress=round(fraction, 3), step=words),
                                    lambda: _exports.get(job, {}).get("cancel"))
            tell(done=True, file=file, subfolder=ROOT + "/exports", progress=1.0)
        except edit_media.Cancelled:
            tell(cancelled=True)
        except Exception as error:
            LOG.exception("MiniMax H3 Vision couldn't export the edit")
            tell(error=str(error))
        finally:
            _exports.pop(job, None)

    loop.create_task(run())
    return web.json_response({"job": job, "file": file, "subfolder": ROOT + "/exports"})


def free_name(folder, stem, ext):
    file, n = stem + ext, 2
    while os.path.exists(os.path.join(folder, file)):
        file = "%s_%d%s" % (stem, n, ext)
        n += 1
    return file


async def export_project(request):
    """The edit and the videos it uses as one zip in minimax_flow/exports, to keep or to open elsewhere."""
    counts, file = await pack_project()
    return web.json_response(dict(counts, file=file, subfolder=ROOT + "/exports"))


async def pack_project():
    async with _lock:
        data = read_project()
    if not data["edit"]["pieces"]:
        raise web.HTTPBadRequest(text="The edit is empty.")
    stem = re.sub(r"[^a-z0-9]+", "_", data["edit"]["name"].lower()).strip("_")[:60] or "edit"
    folder = root_dir("exports")
    file = free_name(folder, stem + "_project", ".zip")
    clips = {c["id"]: c for c in data["clips"] if c.get("kind", "video") == "video" and c.get("file")}
    try:
        counts = await asyncio.to_thread(edit_project.pack, data["edit"], clips, clip_path, os.path.join(folder, file),
                                         data["images"], image_path)
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't save the edit project")
        raise web.HTTPBadRequest(text="Couldn't save the project: %s" % error)
    return counts, file


async def export_clip(request):
    """One video and its reference pictures as a zip in minimax_flow/exports."""
    counts, file = await pack_clip(request.match_info["id"])
    return web.json_response(dict(counts, file=file, subfolder=ROOT + "/exports"))


async def pack_clip(clip_id):
    async with _lock:
        data = read_project()
    clip = next((c for c in data["clips"] if c["id"] == clip_id), None)
    if not clip or clip.get("kind", "video") != "video" or not clip.get("file"):
        raise web.HTTPBadRequest(text="Only a finished video can be saved as a zip.")
    folder = root_dir("exports")
    file = free_name(folder, os.path.splitext(clip["file"])[0], ".zip")
    try:
        counts = await asyncio.to_thread(edit_project.pack_clip, clip, clip_path, os.path.join(folder, file),
                                         data["images"], image_path)
    except edit_project.Bad as error:
        raise web.HTTPBadRequest(text=str(error))
    except Exception as error:
        LOG.exception("MiniMax H3 Vision couldn't save the video as a zip")
        raise web.HTTPBadRequest(text="Couldn't save the zip: %s" % error)
    return counts, file


async def import_project(request):
    """Open a zip: a single video's or an edit project's. Its videos join the app (Video gallery and Edit
    tab) and its reference pictures the Add image library. A project's timeline comes back as "edit",
    for the Edit tab to use; the edit kept here only changes when it asks (?timeline=1)."""
    reader = await request.multipart()
    part = await reader.next()
    while part is not None and not part.filename:
        part = await reader.next()
    if part is None:
        raise web.HTTPBadRequest(text="No file came with the import.")
    temp = root_dir("exports", ".project_%s.zip" % uuid.uuid4().hex[:8])
    try:
        with open(temp, "wb") as f:
            while True:
                chunk = await part.read_chunk(4 << 20)
                if not chunk:
                    break
                f.write(chunk)
        try:
            result = await open_zip(temp, part.filename, request.query.get("timeline") == "1")
        except edit_project.Bad as error:
            raise web.HTTPBadRequest(text=str(error))
    finally:
        try:
            os.remove(temp)
        except OSError:
            pass
    return web.json_response(result)


async def open_zip(temp, filename, timeline=False):
    """A zip's videos and pictures into the app; edit_project.Bad when it isn't one of ours."""
    async with _lock:
        data = read_project()
        existing = {c["id"]: c for c in data["clips"]}
        edit, added, same, new_images, same_images = await asyncio.to_thread(
            edit_project.unpack, temp, existing, root_dir("clips"), clip_path, data["images"], root_dir("images"), image_path)
        now = time.time()
        for i, clip in enumerate(added):
            clip["subfolder"] = ROOT + "/clips"
            clip["imported"] = filename
            # The grid shows the zip's last clip first (the newest at the top); the Edit tab's media
            # panel keeps the zip's order, first clip first ("pool").
            clip["created"] = now - (len(added) - 1 - i) * 0.001
            clip["pool"] = now - i * 0.001
        data["clips"] = added[::-1] + data["clips"]
        for image in new_images:
            image["added"] = now
        data["images"].extend(new_images)
        if edit is not None and timeline:
            data["edit"] = edit
        write_project(data)
    for clip in added:
        broadcast(clip)
    return {"edit": edit, "clips": added, "same": same, "images": new_images, "sameImages": same_images}


# ---- Google Drive: projects saved to and opened from a folder (GDRIVE_FOLDER, GDRIVE_TOKEN) ----
def drive_ready():
    why = gdrive.problem()
    if why == "off":
        raise web.HTTPNotFound(text="Google Drive isn't set up: set GDRIVE_FOLDER and GDRIVE_TOKEN on the pod.")
    if why:
        raise web.HTTPBadRequest(text=why)


async def drive_state(request):
    why = gdrive.problem()
    return web.json_response({"on": why != "off", "problem": None if why == "off" else why, "jobs": gdrive.DRIVE.status()})


async def drive_files(request):
    drive_ready()
    try:
        files = await gdrive.DRIVE.files(fresh=request.query.get("fresh") == "1")
    except RuntimeError as error:
        raise web.HTTPBadRequest(text=str(error))
    return web.json_response({"files": files})


async def drive_save(request):
    """POST {} saves the edit project; {"clip": id} one video's zip."""
    drive_ready()
    body = await request.json()
    clip_id = body.get("clip")

    async def make():
        try:
            _, file = await (pack_clip(clip_id) if clip_id else pack_project())
        except web.HTTPException as error:
            raise RuntimeError(error.text)
        return os.path.join(root_dir("exports"), file)

    if not clip_id:
        async with _lock:
            if not read_project()["edit"]["pieces"]:
                raise web.HTTPBadRequest(text="The edit is empty.")
    job = gdrive.DRIVE.add("save", "", make=make)
    return web.json_response({"job": job, "jobs": gdrive.DRIVE.status()})


async def drive_open(request):
    drive_ready()
    body = await request.json()
    name = os.path.basename(str(body.get("name") or ""))
    if not name.lower().endswith(".zip"):
        raise web.HTTPBadRequest(text="Only .zip files can be opened.")

    async def opener(path, filename):
        try:
            return await open_zip(path, filename)
        except edit_project.Bad as error:
            raise RuntimeError(str(error))

    job = gdrive.DRIVE.add("open", name, temp=root_dir("exports", ".drive_%s.zip" % uuid.uuid4().hex[:8]), opener=opener)
    return web.json_response({"job": job, "jobs": gdrive.DRIVE.status()})


async def drive_cancel(request):
    gdrive.DRIVE.cancel(request.match_info["id"])
    return web.json_response({"jobs": gdrive.DRIVE.status()})


async def cancel_export(request):
    job = _exports.get(request.match_info["job"])
    if job:
        job["cancel"] = True
    return web.json_response({"ok": True})


ROUTES = (
    ("GET", "/mmflow/app", page),
    ("GET", "/mmflow/api/state", get_state),
    ("PUT", "/mmflow/api/ui", put_ui),
    ("GET", "/mmflow/api/loras", get_loras),
    ("POST", "/mmflow/api/images", upload_images),
    ("POST", "/mmflow/api/masks", upload_mask),
    ("GET", "/mmflow/api/models", get_models),
    ("POST", "/mmflow/api/preview", preview),
    ("GET", "/mmflow/api/refmods", refmods),
    ("POST", "/mmflow/api/refmods", refmods),
    ("GET", "/mmflow/api/prompts", prompts),
    ("PUT", "/mmflow/api/prompts", prompts),
    ("POST", "/mmflow/api/prompts/sync", prompts),
    ("POST", "/mmflow/api/models-token", models_token),
    ("DELETE", "/mmflow/api/models-token", models_token),
    ("POST", "/mmflow/api/lora-upload", lora_upload),
    ("PUT", "/mmflow/api/lora-upload/{id}", lora_upload),
    ("POST", "/mmflow/api/lora-upload/{id}/finish", lora_upload),
    ("DELETE", "/mmflow/api/lora-upload/{id}", lora_upload),
    ("GET", "/mmflow/api/lora-links", lora_links),
    ("POST", "/mmflow/api/lora-links", lora_links),
    ("POST", "/mmflow/api/lora-links/{id}/cancel", lora_links),
    ("POST", "/mmflow/api/models/{id}/{action}", models_action),
    ("POST", "/mmflow/api/images/from-clip", image_from_clip),
    ("PATCH", "/mmflow/api/images/{id}", rename_image),
    ("POST", "/mmflow/api/generate", generate),
    ("GET", "/mmflow/api/faces", faces),
    ("POST", "/mmflow/api/clips/{id}/faces", faces),
    ("GET", "/mmflow/api/swap", swaps),
    ("POST", "/mmflow/api/clips/{id}/swap", swaps),
    ("POST", "/mmflow/api/clips/{id}/swap-mark", swap_mark),
    ("GET", "/mmflow/api/swap-mark/{pid}", swap_mark),
    ("POST", "/mmflow/api/import", import_clip),
    ("PATCH", "/mmflow/api/clips/{id}", update_clip),
    ("DELETE", "/mmflow/api/clips/{id}", delete_clip),
    ("PUT", "/mmflow/api/edit", put_edit),
    ("GET", "/mmflow/api/strips/{id}", clip_strip),
    ("POST", "/mmflow/api/export", start_export),
    ("DELETE", "/mmflow/api/export/{job}", cancel_export),
    ("POST", "/mmflow/api/edit/export-project", export_project),
    ("GET", "/mmflow/api/gdrive", drive_state),
    ("GET", "/mmflow/api/gdrive/files", drive_files),
    ("POST", "/mmflow/api/gdrive/save", drive_save),
    ("POST", "/mmflow/api/gdrive/open", drive_open),
    ("POST", "/mmflow/api/gdrive/{id}/cancel", drive_cancel),
    ("POST", "/mmflow/api/edit/import-project", import_project),
    ("POST", "/mmflow/api/clips/{id}/export", export_clip),
    ("POST", "/mmflow/api/import-zip", import_project),
)


def install():
    from server import PromptServer

    routes = PromptServer.instance.routes
    for method, path, handler in ROUTES:
        routes.route(method, path)(handler)
    routes.static("/mmflow/app/", APP_DIR)
