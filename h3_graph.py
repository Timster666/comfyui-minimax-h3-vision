"""MiniMax H3 video graphs, built here without the H3 editor.

Adapted from the H3 editor (comfyui-h3-editor 0.6.1, GPL-3.0, see NOTICE): its checkpoint families
and file picking (project.py), the prompt H3 expects (prompting.py) and the one-pass "draft" graph
of render_graph(). Changed for MiniMax H3 Vision: only what a single prompt with @images needs. The
editor's H3 Take node becomes ComfyUI's own MiniMaxH3ReferenceToVideo or MiniMaxH3ImageToVideo,
which it called with the same inputs for such a clip, and its other nodes are this pack's MMVision
nodes (vision_nodes.py). No GGUF files, no timeline, no finals with a refine pass.

No ComfyUI import at the top, so the graph can be built and checked anywhere.
"""

import fnmatch
import os
import re

ROOT = "minimax_flow"
FPS = 24
SAMPLER_NODE = "15"     # SamplerCustomAdvanced: its progress is the render's steps
SAVE_NODE = "19"        # SaveVideo
LORA_NODE = 100         # the clip's own LoRAs are nodes 100, 101, ...
LORA_LIMIT = 8
LORA_RANGE = (-2.0, 2.0)
H3_PICTURES = 9         # H3 takes at most nine pictures
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0

# H3's text encoder and VAEs are published in several precisions: the first one installed is used.
MODEL_CHOICES = {
    "clip": ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
             "qwen3vl_32b_minimax_h3_bf16.safetensors", "qwen3vl_32b_minimax_h3*"],
    "vae": ["minimax_h3_video_vae_fp16.safetensors", "minimax_h3_video_vae_int8_convrot.safetensors", "minimax_h3_video_vae*"],
    "audio_vae": ["minimax_h3_audio_vae_fp32.safetensors", "minimax_h3_audio_vae*"],
}
WHERE = {"clip": "models/text_encoders", "vae": "models/vae", "audio_vae": "models/vae"}
WHAT = {"clip": "text encoder", "vae": "video VAE", "audio_vae": "audio VAE"}

# Checkpoint families. unet names the ref2va file and unet_text the fl2va file; each is a list of
# names the file may have, and a name ending in * matches any file it starts. A family with no fl2va
# renders prompt-only clips on its ref2va. A HyperFlow file (see is_hyperflow) loads with this pack's
# loader and its own sigmas; any other file loads with ComfyUI's loader and the simple scheduler.
# lora is a LoRA the family always samples with, and its strength. steps is the recommended step count
# (the H3 editor's for a final's first pass).
PROFILES = {
    "hyperflow": {
        "title": "HyperFlow H3 Flash",
        "unet": ["hyperflow/minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors", "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot*"],
        "unet_text": ["hyperflow/minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors", "minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot*"],
        "steps": 6,
        "sampler": "euler",
    },
    "h3_turbo": {
        "title": "MiniMax H3 Turbo",
        "unet": ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "unet_text": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
        "lora": ["minimax_h3_turbo_v4_step600_ema.safetensors", 1.0],
        "steps": 6,
        "sampler": "euler",
    },
    "h3": {
        "title": "MiniMax H3",
        "unet": ["minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
        "unet_text": ["minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
        "steps": 20,
        "sampler": "res_multistep",
    },
    "singularity": {
        "title": "Singularity v2",
        "unet": ["Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors"],
        "steps": 20,
        "sampler": "res_multistep",
    },
    "dasiwa": {
        "title": "DaSiWa Hybrid v4",
        "unet": ["dasiwa_minimax_h3_ref2va_v4_pruned_int8.safetensors", "DasiwaMinimaxH3_dasiwaHybridV4_*"],
        "steps": 20,
        "sampler": "res_multistep",
    },
    "dasiwa_turbo": {
        "title": "DaSiWa Hybrid v4 Turbo",
        "unet": ["dasiwa_minimax_h3_ref2va_v4_pruned_turbo_int8.safetensors", "DasiwaMinimaxH3_dasiwaHybridV4Turbo*"],
        "steps": 8,
        "sampler": "euler",
    },
    "dasiwa_v5": {
        "title": "DaSiWa GXS v5 Turbo",
        "unet": ["dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed.safetensors", "dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo*"],
        "steps": 8,
        "sampler": "euler",
    },
    "eros": {
        "title": "10Eros Max beta 7",
        "unet": ["10Eros_Max_h3_8stepTURBO_beta7_fix1-native_int8_protected_edges_HQ.safetensors", "10Eros_Max_h3_8stepTURBO_beta7*"],
        "steps": 8,
        "sampler": "euler",
    },
    # 10Eros Max beta 5: the TURBO hybrid build, made with a different normalization than beta 7. The card's go-to
    # for motion is res_multistep/simple at 6 to 8 steps.
    "eros_b5": {
        "title": "10Eros Max beta 5",
        "unet": ["10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors", "10Eros_Max_h3_TURBO-hybrid_beta5*"],
        "steps": 8,
        "sampler": "res_multistep",
    },
    # 10Eros Max v1.0: beta5, beta7 and Sulphur3 combined, an fl2va hybrid for pictures or first/last frames alike.
    # Its card: 8 to 12 steps with LCM/simple (res_multistep/simple at 6 to 8, Euler/simple at 4 to 8).
    "eros_v1": {
        "title": "10Eros Max v1.0",
        "unet": ["10Eros_Max_8stepTurbo_v1.0_hybrid-int8_HQ.safetensors", "10Eros_Max_8stepTurbo_v1.0_hybrid-int8*"],
        "steps": 8,
        "sampler": "lcm",
    },
    # The same, unquantized (40.2 GB): for a GPU with the memory for it, such as an RTX PRO 6000.
    "eros_v1_full": {
        "title": "10Eros Max v1.0 full",
        "unet": ["10Eros_Max_8stepTurbo_v1.0_hybrid.safetensors"],
        "steps": 8,
        "sampler": "lcm",
    },
    "prism": {
        "title": "PRISM (Singularity x 10Eros Max)",
        "unet": ["PRISM_DMAD_PDMD_int8.safetensors", "PRISM_DMAD_PDMD*"],
        "steps": 8,
        "sampler": "euler",
        "scheduler": "beta57",
    },
}

# The MMH3 Realism Slider LoRA (Civitai 3041489), as the H3 editor ran it on finals: natural, un-airbrushed skin.
# It sits out the steps above start_sigma, which set the composition, and joins for the detail. V1 if V2 isn't there.
REALISM_FILES = ["MMH3-Realism-V2.safetensors", "MMH3-RealismSlider-V1.safetensors"]
REALISM = {"strength": 1.2, "start_sigma": 0.75, "range": [0.0, 3.0]}
REALISM_NODE = "95"


def realism_lora(files=None):
    return pick_file(REALISM_FILES, available() if files is None else files)


# A scheduler the editor makes itself: the beta schedule with these two numbers, built with ComfyUI's own node.
BETA57 = "beta57"
BETA57_SHAPE = (0.5, 0.7)
# HyperFlow samples on its own two-time schedule; there's no scheduler to choose for it.
HYPERFLOW_SCHEDULE = "hyperflow"


class Missing(Exception):
    """A file a video needs isn't installed."""


def available():
    """The model files ComfyUI lists, for finding a family's files."""
    import folder_paths

    names = set()
    for kind in ("diffusion_models", "unet", "loras", "text_encoders", "clip", "vae"):
        if kind in folder_paths.folder_names_and_paths:
            names.update(folder_paths.get_filename_list(kind))
    return names


def pick_file(names, files):
    """The first of these names that is installed, as ComfyUI lists it (a name ending in * matches any file it starts,
    and a name not found as it is matches the same file name in another folder), or None."""
    if not names:
        return None
    listed = {f.replace("\\", "/"): f for f in files}
    for name in names:
        if name.endswith("*"):
            stem = name[:-1]
            found = sorted((f for f in listed if f.rsplit("/", 1)[-1].startswith(stem)), key=lambda f: (f.rsplit("/", 1)[-1], f))
            if found:
                return listed[found[0]]
        elif name in listed:
            return listed[name]
        else:
            # Moved into a folder (or out of one), e.g. loras/x.safetensors -> loras/MiniMax H3/x.safetensors:
            # the same file name anywhere will do, so saved settings and old clips keep working.
            base = name.replace("\\", "/").rsplit("/", 1)[-1]
            found = sorted(f for f in listed if f.rsplit("/", 1)[-1] == base)
            if found:
                return listed[found[0]]
    return None


def is_hyperflow(name):
    """Whether a checkpoint file is HyperFlow H3 Flash: a .safetensors with hyperflow in its file name."""
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return base.endswith(".safetensors") and "hyperflow" in base


def encoders(files):
    """The text encoder and VAEs to use, and what is missing, in words."""
    found = {key: pick_file(names, files) for key, names in MODEL_CHOICES.items()}
    missing = ["the MiniMax H3 %s (%s in %s)" % (WHAT[key], MODEL_CHOICES[key][0], WHERE[key]) for key, f in found.items() if f is None]
    return found, missing


def family_files(profile, files):
    """A family's ref2va file, fl2va file (or None) and family LoRA (or None), and what it is missing."""
    unet = pick_file(profile["unet"], files)
    lora = profile.get("lora")
    lora_file = pick_file([lora[0]], files) if lora else None
    missing = ([] if unet else [profile["unet"][0]]) + ([lora[0]] if lora and not lora_file else [])
    return unet, pick_file(profile.get("unet_text"), files), lora_file, missing


def models(files=None):
    """Every H3 checkpoint family whose files are installed: id, name, recommended steps and the steps offered."""
    files = available() if files is None else files
    _, missing = encoders(files)
    if missing:
        return []
    rows = []
    for name, profile in PROFILES.items():
        unet, _, _, gaps = family_files(profile, files)
        if gaps:
            continue
        steps = int(profile["steps"])
        if is_hyperflow(unet):
            # HyperFlow has a schedule for 4, 6 and 8 steps only.
            rows.append({"id": name, "name": profile["title"], "steps": steps, "minSteps": 4, "maxSteps": 8, "stepSize": 2,
                         "sampler": profile["sampler"], "scheduler": HYPERFLOW_SCHEDULE})
        else:
            rows.append({"id": name, "name": profile["title"], "steps": steps, "minSteps": 1,
                         "sampler": profile["sampler"], "scheduler": profile.get("scheduler") or "simple",
                         "maxSteps": 40 if steps >= 12 else 16, "stepSize": 1})
    return rows


def and_list(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def why_none(files=None):
    """Why no video model can be used, in words."""
    files = available() if files is None else files
    _, missing = encoders(files)
    if missing:
        return "Videos need " + and_list(missing) + "."
    return ("No MiniMax H3 checkpoint is installed. HyperFlow H3 Flash, for example, needs %s in models/diffusion_models/hyperflow."
            % PROFILES["hyperflow"]["unet"][0].split("/")[-1])


def profile_for(unet, lora=None):
    """The family (a model id) that uses this checkpoint file, and this family LoRA (the turbo one) or none."""
    base = lambda f: os.path.basename(str(f or "")).lower()
    rows = []
    for name, profile in PROFILES.items():
        names = list(profile.get("unet") or []) + list(profile.get("unet_text") or [])
        if any(fnmatch.fnmatch(base(unet), base(f)) for f in names):
            rows.append((name, (profile.get("lora") or [None])[0]))
    match = next((r for r in rows if lora and base(r[1]) == base(lora)), None)
    match = match or next((r for r in rows if not r[1]), None) or (rows[0] if rows else None)
    return match[0] if match else None


def frames(seconds):
    """Nearest legal H3 length, at least about a second: H3 takes lengths in steps of 17 frames."""
    count = float(seconds) * FPS
    return 5 + 17 * max(1, int(round((count - 5) / 17.0)))


PICTURE_TAG = re.compile(r"<Picture (\d+)>")


def clean_pictures(ids, images, prompt=""):
    """The Video tab's pictures, in the order of the prompt bar: <Picture 1> is the first. ValueError when one is
    gone, there are too many, or the prompt names a picture that isn't there."""
    by_id = {i["id"]: i for i in images}
    out = []
    for image_id in ids or []:
        image = by_id.get(image_id)
        if not image:
            raise ValueError("Picture %d isn't in Add image any more." % (len(out) + 1))
        if image_id not in [p["id"] for p in out]:
            out.append({"id": image["id"], "name": image["name"]})
    if len(out) > H3_PICTURES:
        raise ValueError("H3 takes at most %d pictures; there are %d. Take some out." % (H3_PICTURES, len(out)))
    for number in sorted({int(n) for n in PICTURE_TAG.findall(prompt)}):
        if number < 1 or number > len(out):
            raise ValueError("The prompt names <Picture %d>, but there %s." % (
                number, "is 1 picture" if len(out) == 1 else "are %d pictures" % len(out)))
    return out


def mentions(text):
    return re.findall(r"@([A-Za-z0-9_]+)", text)


def sigmas_node(scheduler, steps):
    """The node that makes a render's sigmas for a scheduler by name."""
    if scheduler == BETA57:
        return {"class_type": "BetaSamplingScheduler", "inputs": {
            "model": ["5", 0], "steps": steps, "alpha": BETA57_SHAPE[0], "beta": BETA57_SHAPE[1]}}
    return {"class_type": "BasicScheduler", "inputs": {"model": ["5", 0], "scheduler": scheduler or "simple", "steps": steps, "denoise": 1.0}}


def build_graph(clip, images, files=None, name=None):
    """The ComfyUI graph for one MiniMax H3 Vision clip.

    clip: the clip as MiniMax H3 Vision keeps it (prompt, profile, steps, width, height, len, seed, loras).
    images: every uploaded image, so @names in the prompt can be found.
    """
    files = available() if files is None else files
    found, missing = encoders(files)
    profile = PROFILES.get(clip.get("profile"))
    if profile is None:
        raise Missing("%s isn't a MiniMax H3 model this version knows." % clip.get("model", clip.get("profile")))
    unet_ref, unet_text, family_lora, gaps = family_files(profile, files)
    if missing or gaps:
        raise Missing("%s needs %s." % (profile["title"], and_list(missing + gaps)))

    verbatim = clip.get("pictures") is not None
    if verbatim:
        # The Video tab's pictures, in their order: the prompt says <Picture N> and defines its own subjects.
        by_id = {image["id"]: image for image in images}
        cast = [by_id[p["id"]] for p in clip["pictures"] if p["id"] in by_id][:H3_PICTURES]
    else:
        # An older clip: the @images in the prompt, in the order they were uploaded, one subject with one picture each.
        by_name = {image["name"].lower(): image for image in images}
        mentioned = []
        for word in mentions(clip["prompt"]):
            image = by_name.get(word.lower())
            if image and image["id"] not in mentioned:
                mentioned.append(image["id"])
        cast = [image for image in images if image["id"] in mentioned][:H3_PICTURES]

    # A family's fl2va model takes a prompt alone. Anything with pictures, or with RefMods (references too, which
    # only the ref2va model reads), goes to its ref2va model.
    unet = unet_ref if cast or clip.get("refmods") or not unet_text else unet_text
    text_only = unet == unet_text and not cast
    if cast and verbatim:
        prompt = clip["prompt"].strip()
    elif cast:
        prompt = compose_reference_prompt("", clip["prompt"], subjects=[{"name": i["name"], "picture_count": 1} for i in cast])
    else:
        prompt = compose_prompt("", clip["prompt"])
    width, height, length, steps = int(clip["width"]), int(clip["height"]), frames(clip["len"]), int(clip["steps"])

    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": found["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": found["vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": found["audio_vae"]}},
        "5": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["1", 0], "shift_video": SHIFT_VIDEO, "shift_audio": SHIFT_AUDIO}},
        "9": {"class_type": "MMVisionLivePreview", "inputs": {"model": ["5", 0], "audio_vae": ["4", 0], "folder": ROOT + "/preview"}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": clip.get("sampler") or profile["sampler"]}},
        "12": sigmas_node(clip.get("scheduler") or profile.get("scheduler"), steps),
        "13": {"class_type": "BasicGuider", "inputs": {"model": ["9", 0], "conditioning": ["10", 0]}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(clip["seed"])}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["4", 0]}},
        "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": float(FPS), "audio": ["17", 0]}},
        "19": {"class_type": "SaveVideo", "inputs": {
            "video": ["18", 0], "filename_prefix": "%s/clips/%s" % (ROOT, name or clip["id"]), "format": "auto", "codec": "auto"}},
    }
    if text_only:
        graph["10"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": width, "height": height, "length": length}}
    else:
        graph["10"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": prompt, "width": width, "height": height,
            "length": length, "ref_image_size": "match"}}
        for index, image in enumerate(cast):
            node = str(50 + index)
            graph[node] = {"class_type": "MMVisionLoadStill", "inputs": {"file": "%s/images/%s" % (ROOT, image["file"])}}
            graph["10"]["inputs"]["ref_images.ref_image_%s" % index] = [node, 0]
    if is_hyperflow(unet):
        # Few-step and CFG-free, on its own two-time schedule.
        graph["1"] = {"class_type": "MMVisionHyperFlowLoader", "inputs": {"unet_name": unet}}
        graph["12"] = {"class_type": "MMVisionHyperFlowSigmas", "inputs": {"model": ["5", 0], "steps": steps}}
    model = ["1", 0]
    if profile.get("lora") and family_lora:
        # A LoRA the family always samples with, such as a turbo LoRA.
        graph["94"] = {"class_type": "MMVisionLora", "inputs": {"model": model, "lora_name": family_lora, "strength": profile["lora"][1]}}
        model = ["94", 0]
    rows = clip_loras(clip, files)
    realism = realism_lora(files) if clip.get("realism") else None
    if realism and not any(os.path.basename(file) in REALISM_FILES for file, _ in rows):
        graph[REALISM_NODE] = {"class_type": "MMVisionLora", "inputs": {
            "model": model, "lora_name": realism, "strength": float(clip["realism"]), "start_sigma": REALISM["start_sigma"]}}
        model = [REALISM_NODE, 0]
    for n, (file, strength) in enumerate(rows):
        node = str(LORA_NODE + n)
        graph[node] = {"class_type": "MMVisionLora", "inputs": {"model": model, "lora_name": file, "strength": strength}}
        model = [node, 0]
    graph["5"]["inputs"]["model"] = model
    if clip.get("refmods"):
        # RefMods (ComfyUI-MiniMaxH3Mod): added to what the guider conditions on.
        from . import refmod_bridge
        refmod_bridge.add_to(graph, clip["refmods"])
    return graph


def clip_loras(clip, files):
    """The clip's LoRAs that are installed and not at zero, in order: (file, strength)."""
    rows = []
    for row in clip.get("loras") or []:
        strength = max(LORA_RANGE[0], min(LORA_RANGE[1], float(row.get("strength", 1.0))))
        file = pick_file([str(row["name"]).replace("\\", "/")], files)
        if file and strength:
            rows.append((file, strength))
    return rows[:LORA_LIMIT]


# ---------------------------------------------------------------- the prompt H3 expects
# From the H3 editor's prompting.py: H3 prompts in the structure from MiniMax's writing guides.

DEFAULT_SOUND = "Quiet natural ambience fits the scene and continues throughout."


VOICE = "<Audio 1>"


def dialogue_line(speaker, line, language="English", voice=False):
    """A spoken line. The identifying phrase sits outside <d>, the words inside it, in their own language.

    The speaker ID follows the identity itself, so "@Ana, off-screen" reads "<Subject 1> (S1), off-screen, says".
    """
    who = (speaker or "The speaker").strip()
    who = who[:1].upper() + who[1:]
    if "(S1)" not in who:
        tag = re.match(r"<Subject \d+>", who)
        who = tag.group(0) + " (S1)" + who[tag.end():] if tag else who + " (S1)"
    lead = who
    if voice:
        lead += ", using the voice timbre referenced from %s," % VOICE
    elif who.split("(S1)", 1)[1].strip():
        lead += ","
    return "%s says: <d>[%s] %s</d>" % (lead, language or "English", line.strip())


def compose_prompt(style, action, camera="", speaker="", dialogue="", sound="", music="", first_frame=False,
                   language="English"):
    """A text or first-frame take: instruction line, then the three core fields."""
    parts = [(style or "Live-action, cinematic").strip().rstrip(",.") + ","]
    parts.append(action.strip())
    if camera.strip():
        parts.append(camera.strip())
    if dialogue.strip():
        parts.append(dialogue_line(speaker, dialogue, language))
    body = "integrated_multimodal_description: [Shot 1] " + " ".join(parts)
    fields = [
        body,
        "overall_soundscape: " + (sound.strip() or DEFAULT_SOUND),
        "non_diegetic_music: " + (music.strip() or "N/A"),
    ]
    if first_frame:
        fields.insert(0, "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.")
    return "\n\n".join(fields)


def tag_subjects(text, subjects):
    """Turn @Name into <Subject N>. Longest names first, so @Old Tom wins over @Old."""
    order = sorted(enumerate(subjects, 1), key=lambda pair: -len(pair[1]["name"]))
    for number, subject in order:
        text = re.sub("@" + re.escape(subject["name"]), "<Subject %s>" % number, text, flags=re.IGNORECASE)
    return text


def compose_reference_prompt(style, action, subjects=(), first_frame=False, last_frame=False,
                             camera="", speaker="", dialogue="", sound="", music="", language="English", voice=None):
    """A take with references: the six sections of the full-reference guide.

    Picture numbers follow H3 Take: first frame, last frame, then references.
    subjects are dicts with a name and a description. voice says whose timbre <Audio 1> carries:
    {"subject": N} for a subject in the shot, or {"description": "..."} for an unseen voice.
    """
    subjects = list(subjects)
    picture = 0
    first_tag = last_tag = None
    if first_frame:
        picture += 1
        first_tag = "<Picture %s>" % picture
    if last_frame:
        picture += 1
        last_tag = "<Picture %s>" % picture

    definitions = []
    retention = []
    if first_tag:
        definitions.append("%s is the first frame of [Shot 1]." % first_tag)
        retention.append("%s ([Shot 1] first frame): fully_preserved - the shot begins exactly on this frame." % first_tag)
    if last_tag:
        definitions.append("%s is the last frame of [Shot 1]." % last_tag)
        retention.append("%s ([Shot 1] last frame): fully_preserved - the shot ends exactly on this frame." % last_tag)
    for number, subject in enumerate(subjects, 1):
        # A subject with several pictures names them all, in the order H3 Take is given them.
        tags = []
        for _ in range(max(1, int(subject.get("picture_count") or 1))):
            picture += 1
            tags.append("<Picture %s>" % picture)
        where = tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]
        who = (subject.get("description") or subject["name"]).strip().rstrip(".")
        definitions.append("<Subject %s> is %s in %s." % (number, who, where))
        retention.append("<Subject %s> (appears in [Shot 1]): fully_preserved - their identity and appearance are retained." % number)

    voiced = bool(voice) and bool(dialogue.strip())
    if voiced:
        owner = ("<Subject %s>" % voice["subject"]) if voice.get("subject") else (voice.get("description") or "the speaker").strip().rstrip(".")
        owner = owner if owner.startswith("<") else owner[:1].lower() + owner[1:]
        definitions.append("%s is the voice-timbre reference for %s (S1)." % (VOICE, owner))
        retention.append("%s: reference - the target speaker follows %s's voice timbre and delivery without copying the original signal." % (VOICE, VOICE))

    tasks = []
    if first_tag or last_tag:
        tasks.append("keyframe completion")
    if subjects:
        tasks.append("reference generation")
    if voiced:
        tasks.append("audio reference")
    body = tag_subjects(action.strip(), subjects)
    missing = [
        "<Subject %s>" % number for number, _ in enumerate(subjects, 1)
        if "<Subject %s>" % number not in body
    ]
    lead = ""
    if missing:
        lead = " and ".join(missing) + (" appears in the shot. " if len(missing) == 1 else " appear in the shot. ")
    shot = lead + body
    if first_tag:
        shot = "The shot begins from %s. %s" % (first_tag, shot)
    if camera.strip():
        shot += " " + camera.strip()
    if dialogue.strip():
        shot += " " + dialogue_line(tag_subjects(speaker, subjects), dialogue, language, voice=voiced)
    if last_tag:
        shot += " The shot ends on %s." % last_tag

    sentence = re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0]
    look = (style or "Live-action, cinematic").strip().rstrip(",.")
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(definitions),
        "summary:\n[%s] %s" % (" + ".join(tasks), sentence),
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\nThe target video is in a %s style.\n[Shot 1] %s" % (look.lower(), shot),
        "overall_soundscape:\n" + (sound.strip() or DEFAULT_SOUND),
        "non_diegetic_music:\n" + (music.strip() or "N/A"),
    ])
