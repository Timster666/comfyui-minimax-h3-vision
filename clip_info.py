"""Prompt and settings saved inside each finished file, and read back from files people import.

Every render sends its settings to ComfyUI as extra_pnginfo["mmflow"]. ComfyUI's SaveImage writes
that into the PNG's text chunks and SaveVideo into the MP4's metadata, next to the "prompt" (the
graph) ComfyUI always adds. Reading a file back uses "mmflow" when it is there. Older files have
only the graph, so their settings are worked out from it: MOSS-Image graphs, and the H3 editor's
graphs (MiniMax H3 Vision's own clips, older ones made through the editor, and the H3 editor's clips).
"""

import json
import os
import re

KEY = "mmflow"
VERSION = 1
FIELDS = ("kind", "prompt", "model", "profile", "steps", "cfg", "res", "aspect", "len", "width", "height", "seed", "loras",
          "sampler", "scheduler", "negative", "family", "size", "slots", "edit", "mode", "sheet", "enhance", "sent", "swap",
          "pictures", "refmods", "faces", "realism", "swap", "motion", "silent")
VIDEO_TYPES = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}


def record(clip):
    """What a render saves about itself."""
    out = {"app": "MiniMax H3 Vision", "version": VERSION}
    out.update({k: clip[k] for k in FIELDS if k in clip})
    return out


def extra_data(clip, base=None):
    """ComfyUI's extra_data for a render, so the saved file carries its prompt and settings."""
    extra = dict(base or {})
    extra["extra_pnginfo"] = {KEY: record(clip)}
    return extra


def slug(prompt, most=40):
    """The start of a prompt as a file name: lower case words joined by _, cut at a whole word."""
    words = re.findall(r"[a-z0-9]+", (prompt or "").lower().replace("'", ""))
    out = ""
    for word in words:
        if out and len(out) + 1 + len(word) > most:
            break
        out = (out + "_" + word) if out else word[:most]
    return out or "untitled"


def file_stem(clip):
    return "vision_" + slug(clip.get("prompt"))


# ---------------------------------------------------------------- reading

def tags(path):
    """The text metadata in an image or a video, as a dict of strings."""
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_TYPES:
        try:
            import av
        except ImportError:
            return {}
        with av.open(path) as container:
            found = dict(container.metadata or {})
            for stream in container.streams:
                for key, value in (stream.metadata or {}).items():
                    found.setdefault(key, value)
            return found
    from PIL import Image

    with Image.open(path) as image:
        found = {}
        for key, value in dict(getattr(image, "text", None) or image.info or {}).items():
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            if isinstance(value, str):
                found[key] = value
        return found


def _json(value):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def settings(path):
    """The prompt and settings a file was made with: a dict, with "from" saying how they were found, or None."""
    found = tags(path)
    mine = _json(found.get(KEY))
    if isinstance(mine, dict) and mine.get("prompt"):
        out = {k: mine[k] for k in FIELDS if k in mine}
        out["from"] = "saved"
        return out
    graph = _json(found.get("prompt"))
    if isinstance(graph, dict):
        out = from_graph(graph)
        if out:
            out["from"] = "graph"
            return out
    return None


def _nodes(graph, *names):
    return [(k, n) for k, n in graph.items() if isinstance(n, dict) and n.get("class_type", "").split(".")[-1] in names]


def _literal(value):
    return value if not (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)) else None


def from_graph(graph):
    """Settings from a ComfyUI graph: MOSS-Image, Klein 9B or the H3 editor's, else None."""
    if _nodes(graph, "H3EditorTake", "H3Take", "MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo"):
        return _from_h3(graph)
    if any(n.get("inputs", {}).get("type") == "moss_image" for _, n in _nodes(graph, "CLIPLoader")):
        return _from_moss(graph)
    if any(n.get("inputs", {}).get("type") == "flux2" for _, n in _nodes(graph, "CLIPLoader")) and \
            any("klein" in str(n.get("inputs", {}).get("unet_name", "")).lower() for _, n in _nodes(graph, "UNETLoader")):
        return _from_klein(graph)
    return None


def _loras(graph, skip=()):
    out = []
    for key, node in sorted(_nodes(graph, "LoraLoaderModelOnly", "H3EditorLora", "H3Lora", "MMVisionLora"), key=lambda kn: int(kn[0]) if kn[0].isdigit() else 0):
        if key in skip:
            continue
        inputs = node.get("inputs", {})
        if _literal(inputs.get("lora_name")):
            out.append({"name": inputs["lora_name"], "strength": float(inputs.get("strength_model", inputs.get("strength", 1.0)))})
    return out


def _from_moss(graph):
    sampler = next((n for _, n in _nodes(graph, "KSampler")), None)
    if not sampler:
        return None
    s = sampler["inputs"]
    positive = graph.get(str((s.get("positive") or [""])[0]), {}).get("inputs", {})
    unet = next((n["inputs"].get("unet_name") for _, n in _nodes(graph, "UNETLoader")), None)
    latent = next((n["inputs"] for _, n in _nodes(graph, "EmptyFlux2LatentImage", "EmptyLatentImage", "EmptySD3LatentImage")), {})
    edit = any(True for _ in _nodes(graph, "TextEncodeMOSSImageEdit"))
    # An edit (TextEncodeMOSSImageEdit) has its prompt in "prompt", sizes from the picture, and the edit LoRA
    # isn't one of yours; the picture itself isn't in the file, so the edit comes back without it.
    width, height = _literal(latent.get("width")), _literal(latent.get("height"))
    found = {"kind": "image", "prompt": _literal(positive.get("text", positive.get("prompt"))) or "", "profile": unet,
             "steps": _literal(s.get("steps")), "cfg": _literal(s.get("cfg")), "seed": _literal(s.get("seed")),
             "width": width if isinstance(width, int) else None, "height": height if isinstance(height, int) else None,
             "loras": [l for l in _loras(graph) if "moss_image_edit_lora" not in l["name"]]}
    if edit:
        found["edit"] = {"id": None, "name": "the picture it was made from"}
    return found


def _from_h3(graph):
    take = _nodes(graph, "H3EditorTake", "H3Take", "MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo")[0][1]["inputs"]
    text = _literal(take.get("prompt")) or ""
    steps = next((n["inputs"].get("steps") for _, n in _nodes(graph, "H3EditorHyperFlowSigmas", "H3HyperFlowSigmas", "MMVisionHyperFlowSigmas",
                                                         "BasicScheduler", "BetaSamplingScheduler")), None)
    seed = next((n["inputs"].get("noise_seed") for _, n in _nodes(graph, "RandomNoise")), None)
    unet = next((n["inputs"].get("unet_name") for _, n in _nodes(graph, "UNETLoader", "H3EditorHyperFlowLoader", "H3HyperFlowLoader",
                                                                "MMVisionHyperFlowLoader")), None)
    fps = next((n["inputs"].get("fps") for _, n in _nodes(graph, "CreateVideo")), 24) or 24
    frames = _literal(take.get("length"))
    family_lora = (graph.get("94") or {}).get("inputs", {}).get("lora_name")
    return {"kind": "video", "prompt": h3_prompt(text), "unet": unet, "unet_lora": family_lora, "steps": _literal(steps), "seed": _literal(seed),
            "width": _literal(take.get("width")), "height": _literal(take.get("height")),
            "len": max(1, round(frames / float(fps))) if isinstance(frames, (int, float)) else None,
            # Node 94 is the editor's own LoRA for a checkpoint family (the turbo LoRA), not one the user picked.
            "loras": _loras(graph, skip=("94",))}


def h3_prompt(text):
    """The words someone typed, from the H3 editor's long structured prompt, with <Subject N> back as @names."""
    names = dict(re.findall(r"<Subject (\d+)> is ([A-Za-z0-9_]+) in <Picture \d+>", text))
    shot = re.search(r"^(?:integrated_multimodal_description:\s*)?\[Shot 1\]\s*(.+)", text, re.M)
    summary = re.search(r"summary:\s*\n(.+)", text)
    line = shot.group(1) if shot else summary.group(1) if summary else text
    line = re.sub(r"^\[[a-z ]+generation\]\s*", "", line.strip())
    # Text-only takes start with the editor's style words, which it adds by itself.
    line = re.sub(r"^Live-action, cinematic,\s*", "", line)
    return re.sub(r"<Subject (\d+)>", lambda m: "@" + names[m.group(1)] if m.group(1) in names else m.group(0), line).strip()


def _from_klein(graph):
    """A Klein graph (MiniMax H3 Vision's or ComfyUI's Klein templates): the prompt is the text the guider's
    positive side starts from, through any ReferenceLatent nodes."""
    guider = next((n["inputs"] for _, n in _nodes(graph, "CFGGuider")), None)
    if not guider:
        return None
    link, text = guider.get("positive"), ""
    for _ in range(12):
        node = graph.get(str((link or [""])[0]), {}) if isinstance(link, list) else {}
        if node.get("class_type") == "ReferenceLatent":
            link = node.get("inputs", {}).get("conditioning")
            continue
        text = _literal(node.get("inputs", {}).get("text")) or ""
        break
    unet = next((n["inputs"].get("unet_name") for _, n in _nodes(graph, "UNETLoader")), None)
    sched = next((n["inputs"] for _, n in _nodes(graph, "Flux2Scheduler")), {})
    latent = next((n["inputs"] for _, n in _nodes(graph, "EmptyFlux2LatentImage")), {})
    noise = next((n["inputs"] for _, n in _nodes(graph, "RandomNoise")), {})
    return {"kind": "image", "family": "klein", "prompt": text, "profile": unet, "steps": _literal(sched.get("steps")),
            "cfg": _literal(guider.get("cfg")), "seed": _literal(noise.get("noise_seed")),
            "width": _literal(latent.get("width")), "height": _literal(latent.get("height")), "loras": _loras(graph)}
