"""Swap: Alissonerdx's Character Swap v1.1 and Head Swap v1.0 LoRAs (https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI)
on a finished or imported video, the way the published native-node workflow runs them.

The video is the guide (MiniMaxH3AddGuide at frame 0, with its sound, for the lips), the picture is a native reference
(MiniMaxH3ReferenceToVideo, on a grey background when BiRefNet is installed), and MiniMax H3's own ref2va model samples
with the DMD 8-step turbo LoRA and the swap LoRA: 8 steps, Euler, CFG 1. The prompt is the published template; its one
gap (the character, or the face) is written by the person, or by MOSS-Image's Qwen3.5 9B looking at the picture.
The templates and the describe instructions are Henge's (github.com/alisson-anjos/Henge, MIT, backend/edit_modes.py).
The result is a new clip with the video's own sound.
"""

import folder_paths

from . import h3_graph, moss_bridge, ph3_bridge

TURBO = "minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors"
SAM = "sam3.1_multiplex_fp16.safetensors"   # Comfy-Org/SAM3.1, in models/checkpoints: marks the person to replace
MARK_COLOR, MARK_NAME = "#DC463C", "red"     # Henge's red
BIREFNET = "birefnet.safetensors"
STEPS, SAMPLER = 8, "euler"
MEGAPIXELS = [0.4, 0.6, 0.8, 1.0]
MAX_FRAMES = 361   # 15 s, H3's 17k+5 grid
GREY = 0x808080
REF_MEGAPIXELS = 1.0   # as Henge sends every picture: capped at 1 MP, at H3's largest reference size ("max")
GAP = "{details}"

# Graph nodes added to h3_graph's.
GUIDE, ADD_GUIDE, DESCRIBER, DESCRIBE, FILL = "60", "61", "62", "63", "64"
SAM_LOADER, MARK_NODE, TEMPORAL = "71", "72", "73"
MASK_MODEL, MASK, SIZE, BACKDROP, CUTOUT, SMALL = "65", "66", "67", "68", "69", "70"

CHARACTER_INSTRUCTION = (
    "Analyze the character in this reference image and describe the character's complete visible appearance from head to toe "
    "as one detailed, comma-separated attribute list, in this exact order: 1. gender, apparent height, body type, physique, shoulder width, waist "
    "and body proportions; 2. head and face shape, jawline, facial structure, skin tone, eye color and distinctive facial characteristics; "
    "3. hair color, hairstyle, length, texture, beard, mustache and other facial hair; 4. upper-body clothing with garment types, exact colors, fit, "
    "fabric, textures, collars, sleeves, buttons, zippers, pockets, logos, patterns and layers; 5. lower-body clothing with colors, materials, fit and "
    "design details; 6. footwear; 7. all visible accessories; 8. distinctive features such as tattoos, scars, wrinkles or makeup. "
    "Describe only what is visually identifiable; never invent unseen body parts, clothing or accessories; omit what cannot be determined. "
    "Skip a category that does not apply: write nothing for it, never a \"no ...\" for each missing detail (at most one short phrase such as \"no clothing\"). "
    "Never name or guess real people, celebrities, actors or fictional characters; describe appearance only. "
    "Do not describe the background, camera, lighting or image composition. Start with the gender (\"Male, ...\" or \"Female, ...\"); for an animal, creature or other non-human "
    "character start with what it is instead (e.g. \"Small blue lizard-like creature, ...\"), "
    "with no article before it and no full stop after it. Output only the comma-separated list, on one line, nothing else.")

CHARACTER_TEMPLATE = (
    "body_swap: subject_definitions: <Subject 1> is the {details} shown in <Picture 1>, whose identity, face, head, hair or fur, skin, "
    "body shape, proportions and clothing define the character in the target video. The guide video provides the scene and the performance: its "
    "background, environment, lighting, camera, framing, body pose, hand and arm movement, head rotation, gaze direction, lip movement and facial "
    "expression, frame by frame; only the main subject of the guide video is replaced: its face, hair, body and clothing become <Subject 1>.  "
    "summary: [video generation + reference] Replace the guide subject with <Subject 1>, keeping the scene, camera and the complete motion and "
    "performance of the guide video.  detailed_description: [Shot 1] <Subject 1> takes the exact place of the guide subject and moves with the same "
    "timing, body pose, gestures, head motion, gaze, speech and facial expressions, while keeping the complete identity from <Picture 1>: facial "
    "structure, hair or fur, skin, body shape and clothing stay stable in every frame. The background, lighting, colors and camera of the guide "
    "video are preserved. Prevent identity drift, clothing changes, texture crawling and flicker.")

FACE_INSTRUCTION = (
    'Describe this face in a short comma-separated attribute list, in exactly this style: "Male, oval face shape, '
    'average-sized head with strong jawline, light brown skin, dark eyes, black tousled hair, silver hoop earring." Cover, in order: '
    'gender, head/face shape and proportions (e.g. oval/round/square/heart-shaped, narrow/wide, jaw structure, whether the head reads '
    'as small/average/large relative to the shoulders), skin tone, eye color, hair color and style, and any distinctive features '
    '(facial hair, jewelry, makeup, glasses, etc.). Never name or guess real people, celebrities or fictional characters. Only output the description, nothing else.')

HEAD_TEMPLATE = """head_swap:

subject_definitions:

<Subject 1> is the reference head identity from <Picture 1>, including the complete head, facial identity, facial structure, head shape, hairstyle, hairline, skin appearance, proportions, and all distinctive visual features.

Facial details of <Subject 1>:
{details}

<Subject 2> is the original person in <Video 1>, whose body, neck, clothing, pose, physical proportions below the head, actions, motion, timing, and interaction with the environment must remain unchanged.

<Video 1> is the source video for the target video edit, providing the original body performance, motion, camera movement, framing, timing, lighting, environment, scene composition, and temporal structure.

summary:

[video editing + reference generation]

The target video is an edited version of <Video 1>. Replace only the entire head of <Subject 2> with the reference head identity defined by <Subject 1> from <Picture 1>. Preserve the source video's body, clothing, performance, motion, camera behavior, environment, lighting, timing, and composition.

retention_analysis:

<Subject 1> (appears throughout the target video): attribute_transfer - the complete head identity from <Picture 1>, including facial identity, facial structure, hairstyle, head shape, proportions, skin appearance, and distinctive features, is transferred consistently onto the body of <Subject 2>.

<Subject 2> (appears throughout the target video): partially_preserved - the original body, neck position, clothing, pose, motion, gestures, actions, and physical performance are preserved, while only the original head is replaced.

<Video 1> (entire target timeline): partially_preserved - the original video timing, body motion, camera movement, framing, lighting, background, environment, scene composition, and temporal structure are preserved; only the subject's head appearance is modified.

detailed_description:

Preserve <Video 1> as the complete temporal and compositional foundation of the target video. <Subject 2> retains the exact original body, clothing, pose, body proportions, movement, gestures, actions, speed, timing, and interaction with the environment.

Replace only the entire original head of <Subject 2> with <Subject 1>, the reference head identity from <Picture 1>. Preserve the complete recognizable identity of <Subject 1>, including facial structure, skull and head shape, hairstyle, hairline, skin appearance, facial proportions, eyes, eyebrows, nose, mouth, jawline, ears, and distinctive identifying features.

The transferred head follows the original head motion and performance of <Subject 2> throughout <Video 1>, including head rotation, tilt, pitch, yaw, natural translation, facial orientation, gaze direction, and perspective changes. Maintain anatomically correct head scale and a seamless connection between the transferred head and the original neck.

Preserve the lighting conditions of <Video 1> on the transferred head. The head naturally responds to changes in illumination, shadows, highlights, color temperature, exposure, depth of field, motion blur, occlusion, camera angle, and perspective while continuously maintaining the identity established by <Picture 1>.

Maintain strong temporal consistency across every frame. The identity, facial geometry, hairstyle, head size, skin appearance, and distinguishing features of <Subject 1> remain stable during motion and viewpoint changes. Prevent identity drift, facial morphing, hairstyle changes, head-size fluctuations, geometry instability, texture crawling, temporal flickering, frame-to-frame inconsistency, or sudden changes in facial appearance.

Do not alter the body, clothing, hands, pose, gestures, action, background, objects, environment, camera trajectory, framing, lens behavior, scene timing, or composition inherited from <Video 1>. The final result should appear as though <Subject 1>'s head naturally belonged to <Subject 2>'s body during the original recorded performance.

overall_soundscape:

Preserve the original audiovisual timing and visible action structure of <Video 1>. No visual modifications are made outside the head replacement.

non_diegetic_music:
N/A"""

# A marked person: Henge's single-subject wording after the Character Swap template (edit_modes.shot_swap_plan).
TRACK_IDENTITY = ("Each identity follows its marked person, regardless of screen position, distance from the camera, "
                  "crossings or occlusion. Preserve the guide person's orientation: a back view stays a back view; transfer only "
                  "the reference features visible from that angle. The reference supplies appearance, while the guide supplies "
                  "pose and facing direction. Keep absent people out of the shot.")
MARKED = (" Replace only the %s-marked person; keep every other person unchanged. Remove all temporary color markers and text "
          "tags. %s Do not add people, insets or duplicate characters." % (MARK_NAME, TRACK_IDENTITY))

# Motion Transfer v1.0: the manual prompt of the author's native workflow (minimax_h3_motion_transfer_workflow_native_nodes_v1.0.json),
# to edit: say which picture is who, or what (an outfit, an object, a place, a style).
MOTION_TEMPLATE = """motion_transfer:
subject_definitions:
<Subject 1> is the subject in <Picture 1>; preserve its visible identity, face, hair or fur, body proportions and outfit. Picture roles are defined by this prompt, not by slot number. Additional connected pictures may specify other subjects, outfits, objects, environments or style; edit these sections to assign each actual attached picture explicitly.
summary:
[video generation + reference] Transfer the source performance to <Subject 1>, preserving the action order, timing and camera of the guide.
retention_analysis:
<Subject 1>: preserve the reference identity and appearance throughout the clip. Keep the guide setting unless a connected reference is explicitly assigned to replace it.
detailed_description:
[Shot 1] <Subject 1> follows the guide's motion, poses, gestures, head movement and expression in order and at matching times. Keep identity stable during occlusion, turns and back views. Do not turn a back view into a front view. Several reference images may describe one subject; image count does not imply performer count. Do not copy an identity reference's still pose or backdrop as the motion or setting.
overall_soundscape:
Keep the source soundtrack when connected; do not invent dialogue.
non_diegetic_music:
No additional music requested."""
MOTION_TRIGGER = "motion_transfer:"
# Motion Transfer TARA v1.0: the LoRA the author trained for several people, objects or vehicles, run his way: the
# video as a native <Video 1> reference (no Add Guide), with his temporal alignment patch from ComfyUI-BFSNodes
# (BFSMiniMaxH3NativeTemporal) at the training defaults. H3 takes a reference video of 2 to 15 seconds.
TARA_LORA = "minimax_h3_motion_transfer_tara_v1.0_ref2va.safetensors"
TARA_NODE = "BFSMiniMaxH3NativeTemporal"
TARA_MIN_FRAMES = 56   # the first 17k+5 count of at least 2 s at 24 fps
METHODS = {"tara": "TARA · several people", "guide": "Motion Transfer v1.0"}
MOTION_REF_MEGAPIXELS = 1.0   # the workflow's: each picture capped at 1 MP, sent at H3's largest reference size

MODES = {
    "character": {"name": "Character Swap v1.1", "lora": "minimax_h3_character_swap_v1.1_ref2va_fro.safetensors",
                  "template": CHARACTER_TEMPLATE, "instruction": CHARACTER_INSTRUCTION, "scheduler": "simple"},
    "head": {"name": "Head Swap v1.0", "lora": "minimax_h3_head_swap_v1.0_r32.safetensors",
             "template": HEAD_TEMPLATE, "instruction": FACE_INSTRUCTION, "scheduler": "beta"},
    "motion": {"name": "Motion Transfer v1.0", "lora": "minimax_h3_motion_transfer_v1.0_ref2va.safetensors",
               "template": MOTION_TEMPLATE, "instruction": None, "scheduler": "simple"},
}


def birefnet():
    """The BiRefNet background remover, as ComfyUI lists it, if it's installed (models/background_removal)."""
    try:
        return h3_graph.pick_file([BIREFNET], folder_paths.get_filename_list("background_removal"))
    except KeyError:
        return None


def tara_why(files=None):
    """Why TARA can't run here, in words, or None."""
    files = h3_graph.available() if files is None else files
    lack = []
    if not h3_graph.pick_file([TARA_LORA], files):
        lack.append(TARA_LORA + " (gear menu › Models › Swap and Motion)")
    try:
        import nodes
        if TARA_NODE not in nodes.NODE_CLASS_MAPPINGS:
            lack.append("the ComfyUI-BFSNodes node pack (on RunPod, ENABLE_SWAP=true installs it)")
    except ImportError:
        pass
    return ("TARA needs " + " and ".join(lack) + ".") if lack else None


def sam():
    """SAM 3.1, as ComfyUI lists it in models/checkpoints, if it's installed."""
    return h3_graph.pick_file([SAM], folder_paths.get_filename_list("checkpoints"))


BASE = "h3"   # the author's: MiniMax H3's own ref2va model, with the DMD turbo LoRA


def turbo(profile_id):
    """Whether a model family samples in few steps by itself (a turbo build, or with its own turbo LoRA): a swap
    then runs it its own way, without the DMD turbo LoRA."""
    profile = h3_graph.PROFILES[profile_id]
    return bool(profile.get("lora")) or profile["steps"] <= STEPS


def models(video_models):
    """The installed models a swap can run on: not HyperFlow, whose schedule is its own. The author's first."""
    usable = [m for m in video_models if m.get("scheduler") != h3_graph.HYPERFLOW_SCHEDULE and m["id"] in h3_graph.PROFILES]
    return sorted(usable, key=lambda m: m["id"] != BASE)


def describer(files=None):
    """MOSS-Image's Qwen3.5 9B, which writes the description from the picture, if it's installed."""
    files = h3_graph.available() if files is None else files
    return h3_graph.pick_file(moss_bridge.TEXT_ENCODERS[:1] + [n + ".safetensors" for n in moss_bridge.TEXT_ENCODERS], files)


def status(files=None):
    """What the Swap dialog needs: per mode, why it can't run (or None), and whether a picture can be described."""
    files = h3_graph.available() if files is None else files
    _, missing = h3_graph.encoders(files)
    modes = {}
    for key, mode in MODES.items():
        lack = missing + [f for f in (TURBO, mode["lora"]) if not h3_graph.pick_file([f], files)]
        modes[key] = ("%s needs %s (gear menu › Models › Swap)." % (mode["name"], h3_graph.and_list(lack))) if lack else None
    return {"modes": modes, "methods": {"tara": tara_why(files), "guide": None}, "motionTemplate": MOTION_TEMPLATE, "describe": bool(describer(files)), "megapixels": MEGAPIXELS, "maxSeconds": MAX_FRAMES // 24,
            "background": bool(birefnet()), "mark": bool(sam())}


def template_of(mode, marked=False):
    return MODES[mode]["template"] + (MARKED if marked else "")


def prompt_of(mode, details, marked=False):
    return template_of(mode, marked).replace(GAP, " ".join(details.split()).strip().strip('"').rstrip("."))


def mark_nodes(graph, mark):
    """SAM 3.1 and the node that paints the marked person on the guide's frames."""
    name = sam()
    if not name:
        raise h3_graph.Missing("Picking the person needs SAM 3.1 (sam3.1_multiplex_fp16.safetensors) in models/checkpoints "
                               "(gear menu › Models › Swap).")
    graph[SAM_LOADER] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}}
    graph[MARK_NODE] = {"class_type": "MMVisionMarkPerson", "inputs": {
        "images": [GUIDE, 0], "sam_model": [SAM_LOADER, 0], "x": float(mark["x"]), "y": float(mark["y"]),
        "color": MARK_COLOR, "opacity": float(mark.get("opacity", 1.0))}}
    return [MARK_NODE, 0]


def mark_graph(source, mark, megapixels, prefix):
    """The guide with the person marked, as a video to check before swapping."""
    graph = {GUIDE: {"class_type": "MMVisionLoadGuide", "inputs": {"file": source, "megapixels": float(megapixels), "max_frames": MAX_FRAMES}}}
    marked = mark_nodes(graph, mark)
    graph["18"] = {"class_type": "CreateVideo", "inputs": {"images": marked, "fps": 24.0}}
    graph["19"] = {"class_type": "SaveVideo", "inputs": {"video": ["18", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}}
    return graph


def written_prompt(entry):
    """The prompt with the description the text model wrote, from a finished swap's history, or None."""
    return ph3_bridge.upsampled_text(entry)


def guide_shape(path, megapixels):
    """The width, height and frame count MMVisionLoadGuide makes of a video, worked out the same way."""
    import av
    with av.open(path) as container:
        stream = container.streams.video[0]
        src_w, src_h = stream.codec_context.width, stream.codec_context.height
        seconds = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / 1e6
    length = min(int(seconds * 24), MAX_FRAMES)
    length -= (length - 5) % 17
    scale = (megapixels * 1e6 / (src_w * src_h)) ** 0.5
    return max(32, round(src_w * scale / 32) * 32), max(32, round(src_h * scale / 32) * 32), length


def motion_prompt(prompt):
    """A Motion Transfer prompt as sent: the person's own, with the trigger it was trained on in front."""
    prompt = prompt.strip()
    return prompt if prompt.startswith(MOTION_TRIGGER) else MOTION_TRIGGER + "\n" + prompt


def build(clip, images, source, mode, details="", megapixels=0.6, files=None, name=None, mark=None, model=BASE, method="guide"):
    """The graph that swaps the picture's character (or head) into a video.

    clip: the new clip: the picture as <Picture 1>, the seed, and the prompt (made here from the template).
    source: the video, as a path inside the output folder. details: the character (or face) in words; empty means
    MOSS-Image's text model describes the picture. model: the model family to swap on; the author's is the base
    model with the DMD turbo LoRA (as is any other non-turbo family), and a turbo family runs on its own. mark: {x, y, opacity}, a point on the first frame: only the person
    there is replaced (Character only), painted red on the guide and named in the prompt.
    Motion: clip carries up to 9 pictures, the person's own prompt and LoRAs (details is not used); method "tara" runs
    Motion Transfer TARA v1.0 the author's way, "guide" Motion Transfer v1.0.
    """
    files = h3_graph.available() if files is None else files
    spec = MODES[mode]
    motion = mode == "motion"
    tara = motion and method == "tara"
    if tara and tara_why(files):
        raise h3_graph.Missing(tara_why(files))
    text = describer(files) if not motion and not details.strip() else None
    if not motion and not details.strip() and not text:
        raise h3_graph.Missing("Describe the %s in words: describing the picture needs MOSS-Image's text encoder "
                               "(qwen3.5_9b_w4a8_gptq.safetensors)." % ("character" if mode == "character" else "face"))
    profile = h3_graph.PROFILES[model]
    own = turbo(model)
    clip = dict(clip, profile=model, refmods=None, realism=None,
                steps=profile["steps"] if own else STEPS, sampler=profile["sampler"] if own else SAMPLER,
                scheduler=(profile.get("scheduler") or spec["scheduler"]) if own else spec["scheduler"],
                loras=([] if own else [{"name": TURBO, "strength": 1.0}]) + [{"name": TARA_LORA if tara else spec["lora"], "strength": 1.0}]
                + (list(clip.get("loras") or []) if motion else []),
                prompt=motion_prompt(clip["prompt"]) if motion
                else prompt_of(mode, details, bool(mark)) if details.strip() else template_of(mode, bool(mark)))
    graph = h3_graph.build_graph(clip, images, files=files, name=name)
    picture = "50"
    graph[GUIDE] = {"class_type": "MMVisionLoadGuide", "inputs": {"file": source, "megapixels": float(megapixels), "max_frames": MAX_FRAMES}}
    cond = graph["10"]["inputs"]
    cond.update({"width": [GUIDE, 2], "height": [GUIDE, 3], "length": [GUIDE, 4]})

    if motion:
        # Every picture as the author's workflow sends them: up to 1 MP each, at H3's largest reference size.
        cond["ref_image_size"] = "max"
        for key in [k for k in cond if k.startswith("ref_images.")]:
            small = "8%s" % key.rsplit("_", 1)[1]
            graph[small] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
                "image": cond[key], "upscale_method": "lanczos", "megapixels": MOTION_REF_MEGAPIXELS, "resolution_steps": 32}}
            cond[key] = [small, 0]
    else:
        cond["ref_image_size"] = "max"
        reference = [picture, 0]
        remover = birefnet()
        if remover:
            # The workflow's reference: the character cut out onto grey, so nothing of the picture's background comes along.
            graph[MASK_MODEL] = {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": remover}}
            graph[MASK] = {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": [MASK_MODEL, 0], "image": reference}}
            graph[SIZE] = {"class_type": "GetImageSize", "inputs": {"image": reference}}
            graph[BACKDROP] = {"class_type": "EmptyImage", "inputs": {"width": [SIZE, 0], "height": [SIZE, 1], "batch_size": 1, "color": GREY}}
            graph[CUTOUT] = {"class_type": "ImageCompositeMasked", "inputs": {
                "destination": [BACKDROP, 0], "source": reference, "x": 0, "y": 0, "resize_source": False, "mask": [MASK, 0]}}
            reference = [CUTOUT, 0]
        graph[SMALL] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": reference, "upscale_method": "lanczos", "megapixels": REF_MEGAPIXELS, "resolution_steps": 32}}
        cond["ref_images.ref_image_0"] = [SMALL, 0]

    if text:
        graph[DESCRIBER] = {"class_type": "CLIPLoader", "inputs": {"clip_name": text, "type": "minimax", "device": "default"}}
        graph[DESCRIBE] = {"class_type": "TextGenerateV2", "inputs": {
            "clip": [DESCRIBER, 0], "prompt": spec["instruction"], "images.image1": [picture, 0], "max_length": 1024,
            "sampling_mode": "off", "thinking": False, "use_default_template": True, "mtp": "off"}}
        graph[FILL] = {"class_type": "StringReplace", "inputs": {"string": template_of(mode, bool(mark)), "find": GAP, "replace": [DESCRIBE, 0]}}
        graph[ph3_bridge.UPSAMPLED_NODE] = {"class_type": "PreviewAny", "inputs": {"source": [FILL, 0]}}
        cond["prompt"] = [FILL, 0]

    if tara:
        # The video as <Video 1>, aligned frame by frame by the patch, which the live preview and guider then use.
        cond["ref_videos.ref_video_0"] = [GUIDE, 0]
        graph[TEMPORAL] = {"class_type": TARA_NODE, "inputs": {
            "model": ["5", 0], "motion_video_index": 1, "strength": 2.0, "window_seconds": 0.2, "falloff_seconds": 0.25, "max_penalty": 6.0}}
        graph["9"]["inputs"]["model"] = [TEMPORAL, 0]
        graph.pop("17", None)
        graph["18"]["inputs"]["audio"] = [GUIDE, 1]
        return graph
    guide = mark_nodes(graph, mark) if mark else [GUIDE, 0]
    graph[ADD_GUIDE] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
        "positive": ["10", 0], "latent": ["10", 1], "vae": ["3", 0], "audio_vae": ["4", 0], "image": guide,
        "audio": [GUIDE, 1], "frame_idx": 0}}
    graph["13"]["inputs"]["conditioning"] = [ADD_GUIDE, 0]
    graph.pop("17", None)   # the video's own sound is kept
    graph["18"]["inputs"]["audio"] = [GUIDE, 1]
    return graph
