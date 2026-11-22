"""The graph nodes MiniMax H3 Vision's video renders use, besides ComfyUI's own.

They come from the H3 editor's nodes (see NOTICE), under names of their own (MMVision...), so this pack
and the H3 editor can be installed side by side. Each video graph uses them like this:

- MMVisionLoadStill: an @image, read from the output folder, where the app keeps uploads.
- MMVisionLoadVideo: a clip, read from the output folder, for Refine faces.
- MMVisionLoadGuide: a clip as a swap's guide: 24 fps, H3 size, 17k+5 frames, with its sound.
- MMVisionMarkPeople: people of a guide, each painted in a colour (SAM 3), so a swap or motion knows who becomes whom.
- MMVisionHyperFlowLoader and MMVisionHyperFlowSigmas: HyperFlow H3 Flash, its model and its 4, 6 or 8-step schedule.
- MMVisionLora: an H3 LoRA, renumbered for HyperFlow and added at run time on INT8 weights.
- MMVisionLivePreview: the moving preview of the clip at every step.
"""

import json
import os

import folder_paths
import numpy as np
import torch
from comfy_api.latest import io
from PIL import Image, ImageOps

from . import h3_lora, hyperflow, live_preview

CATEGORY = "MiniMax H3 Vision"


def output_file(file):
    base = os.path.realpath(folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(base, file))
    if not path.startswith(base + os.sep) or not os.path.exists(path):
        raise ValueError("no file at %s in the output folder" % file)
    return path


class LoadStill(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionLoadStill",
            display_name="Vision Load Still",
            category=CATEGORY,
            description="Read a picture from the output folder, where MiniMax H3 Vision keeps uploaded images.",
            inputs=[io.String.Input("file", tooltip="Path inside the output folder.")],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        with Image.open(output_file(file)) as image:
            pixels = np.array(ImageOps.exif_transpose(image).convert("RGB")).astype(np.float32) / 255.0
        return io.NodeOutput(torch.from_numpy(pixels)[None])


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionLoadVideo",
            display_name="Vision Load Video",
            category=CATEGORY,
            description="Read a video from the output folder, where MiniMax H3 Vision keeps its clips.",
            inputs=[io.String.Input("file", tooltip="Path inside the output folder.")],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        from comfy_api.latest import InputImpl
        return io.NodeOutput(InputImpl.VideoFromFile(output_file(file)))


class LoadGuide(io.ComfyNode):
    """A clip as the guide of a swap: at H3's 24 fps, scaled to the megapixels asked for (sides a multiple of 32, the
    clip's own shape), cut to the longest 17k+5 frame count that fits, with its sound cut to match."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionLoadGuide",
            display_name="Vision Load Guide Video",
            category=CATEGORY,
            description="Read a video from the output folder as a MiniMax H3 guide: 24 fps, H3 size, 17k+5 frames.",
            inputs=[io.String.Input("file", tooltip="Path inside the output folder."),
                    io.Float.Input("megapixels", default=0.6, min=0.1, max=2.0, step=0.01),
                    io.Int.Input("max_frames", default=361, min=5, max=3600)],
            outputs=[io.Image.Output(display_name="images"), io.Audio.Output(display_name="audio"),
                     io.Int.Output(display_name="width"), io.Int.Output(display_name="height"),
                     io.Int.Output(display_name="length")],
        )

    @classmethod
    def execute(cls, file, megapixels, max_frames) -> io.NodeOutput:
        from comfy_api.latest import InputImpl
        import comfy.utils
        parts = InputImpl.VideoFromFile(output_file(file)).get_components()
        rate = float(parts.frame_rate) or 24.0
        count = int(parts.images.shape[0] * 24 / rate)
        length = min(count, max_frames)
        length -= (length - 5) % 17
        if length < 5:
            raise ValueError("The video is too short for H3: it needs at least 5 frames at 24 fps.")
        pick = [min(parts.images.shape[0] - 1, int(n * rate / 24)) for n in range(length)]
        frames = parts.images[pick]
        src_h, src_w = frames.shape[1], frames.shape[2]
        scale = (megapixels * 1e6 / (src_w * src_h)) ** 0.5
        width, height = max(32, round(src_w * scale / 32) * 32), max(32, round(src_h * scale / 32) * 32)
        frames = comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        audio = parts.audio
        if audio is not None:
            audio = {"waveform": audio["waveform"][..., :int(length / 24 * audio["sample_rate"])], "sample_rate": audio["sample_rate"]}
        return io.NodeOutput(frames, audio, width, height, length)


class MarkPeople(io.ComfyNode):
    """People in a guide video, each painted in their own colour, so a swap or a motion transfer knows who becomes
    whom: SAM 3 finds each person at a point on the first frame and follows them through the clip. The
    point-to-person logic is Henge's (github.com/alisson-anjos/Henge, MIT, point_tracking.py): of the masks at the
    point, the largest that contains it. A click usually gives just the part clicked (a head), so the whole people
    are found too, as Henge's find_people does (SAM 3's text prompt "person", up to 8), and the person the click is
    in is the one marked. Each mark is grown a little, so no edge of the original person is left to come through.
    Everyone is found on the unpainted frame and followed on their own."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionMarkPeople",
            display_name="Vision Mark People",
            category=CATEGORY,
            description="Paint the people at points of the first frame, each in a colour, through the whole clip, with SAM 3.",
            inputs=[io.Image.Input("images"), io.Model.Input("sam_model"), io.Clip.Input("sam_clip"),
                    io.String.Input("people", default='[{"x": 0.5, "y": 0.5, "color": "#DC463C"}]',
                                    tooltip='JSON: [{"x", "y", "color"}], x and y from 0 to 1 across and down the first frame.'),
                    io.Float.Input("opacity", default=1.0, min=0.1, max=1.0, step=0.05)],
            outputs=[io.Image.Output(display_name="images"), io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, images, sam_model, sam_clip, people, opacity) -> io.NodeOutput:
        from comfy.ldm.sam3.tracker import unpack_masks
        from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack
        n, h, w = images.shape[:3]
        marked, union = images.clone(), torch.zeros(n, h, w, dtype=torch.bool)

        def full_size(found):
            found = found.cpu() > 0.5
            if found.numel() and tuple(found.shape[-2:]) != (h, w):
                found = torch.nn.functional.interpolate(found.float()[:, None], size=(h, w), mode="nearest")[:, 0] > 0.5
            return found

        # Everyone in the first frame, whole (Henge's find_people: SAM 3 keeps one detection per text prompt unless
        # the prompt asks for more).
        cond = sam_clip.encode_from_tokens_scheduled(sam_clip.tokenize("person"))
        cond = [[cond[0][0], {**cond[0][1], "sam3_multi_cond": [
            {"cond": cond[0][0], "attention_mask": cond[0][1].get("attention_mask"), "max_detections": 8}]}]]
        everyone = full_size(SAM3_Detect.execute(model=sam_model, image=images[:1], conditioning=cond, threshold=0.4,
                                                 refine_iterations=1, individual_masks=True)[0])
        grow = max(3, round(0.012 * min(h, w)))
        for number, person in enumerate(json.loads(people), 1):
            px = min(w - 1, max(0, round(float(person["x"]) * w)))
            py = min(h - 1, max(0, round(float(person["y"]) * h)))
            # More decoder refinement drops the point prompt (Henge's note), so one pass.
            found = full_size(SAM3_Detect.execute(model=sam_model, image=images[:1], positive_coords=json.dumps([{"x": px, "y": py}]),
                                                  refine_iterations=1, individual_masks=True)[0])
            part = max([m for m in found if bool(m[py, px])], key=lambda m: int(m.sum()), default=None)
            # The whole person the click is in: the smallest one that holds it when people overlap, else the one
            # that shares the most with what the click found.
            holders = [m for m in everyone if bool(m[py, px])]
            whole = min(holders, key=lambda m: int(m.sum())) if holders else None
            if whole is None and part is not None and len(everyone):
                best = max(everyone, key=lambda m: int((m & part).sum()))
                whole = best if int((best & part).sum()) > 0.5 * int(part.sum()) else None
            if whole is None and part is None:
                raise ValueError("SAM found nobody at person %d's point on the first frame: click inside the person." % number)
            seed = (whole if part is None else part if whole is None else whole | part).float()[None]
            mask = torch.zeros(n, h, w, dtype=torch.bool)
            if n > 1:
                packed = SAM3_VideoTrack.execute(images=images, model=sam_model, initial_mask=seed, max_objects=1)[0].get("packed_masks")
                if packed is not None:
                    followed = unpack_masks(packed).cpu().float()
                    if tuple(followed.shape[-2:]) != (h, w):
                        followed = torch.nn.functional.interpolate(followed, size=(h, w), mode="nearest")
                    mask |= followed[:, 0] > 0.5
            mask[0] = seed[0].bool()
            # A few pixels more all round, so hair, fingers and blurred edges of the original are painted too.
            mask = torch.nn.functional.max_pool2d(mask.float()[:, None], 2 * grow + 1, stride=1, padding=grow)[:, 0] > 0.5
            color = str(person["color"]).lstrip("#")
            rgb = torch.tensor([int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)], dtype=images.dtype)
            marked[mask] = rgb if opacity >= 1 else images[mask] * (1 - opacity) + rgb * opacity
            union |= mask
        return io.NodeOutput(marked, union.float())


class Lora(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionLora",
            display_name="Vision H3 LoRA",
            category=CATEGORY,
            description=(
                "Apply a LoRA trained on MiniMax H3. On INT8 or FP8 weights and HyperFlow's factorized layers it is "
                "added at run time, so small updates are not rounded away. On HyperFlow, whose blocks are pruned, its "
                "blocks move down to the layers they were trained for. Several in a row stack."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.0, min=-4.0, max=4.0, step=0.05),
                io.Float.Input("start_sigma", default=1.0, min=0.0, max=1.0, step=0.005, optional=True,
                               tooltip="Highest noise level (sigma) where this LoRA runs. 1.0 runs from the first step."),
                io.Float.Input("end_sigma", default=0.0, min=0.0, max=1.0, step=0.005, optional=True,
                               tooltip="Lowest noise level (sigma) where this LoRA runs. 0.0 runs to the final step."),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, lora_name, strength, start_sigma=1.0, end_sigma=0.0) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(h3_lora.apply(model, path, lora_name, strength, "auto", start_sigma, end_sigma))


class HyperFlowLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionHyperFlowLoader",
            display_name="Vision HyperFlow Loader",
            category=CATEGORY,
            description=(
                "Load a HyperFlow H3 Flash INT8 safetensors checkpoint: MiniMax H3 distilled to 4, 6 or 8 steps, "
                "with some blocks pruned. Sample it with Vision HyperFlow Sigmas, euler and no CFG."
            ),
            inputs=[io.Combo.Input("unet_name", options=hyperflow.files())],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def validate_inputs(cls, unet_name) -> bool | str:
        # Checked here instead of against the list, so the file may sit in another folder.
        if not isinstance(unet_name, str):
            return True
        try:
            hyperflow.model_path(unet_name)
        except FileNotFoundError:
            base = unet_name.replace("\\", "/").rsplit("/", 1)[-1]
            return "%s is not installed. Put the file in models/diffusion_models/hyperflow, then restart ComfyUI." % base
        return True

    @classmethod
    def execute(cls, unet_name) -> io.NodeOutput:
        return io.NodeOutput(hyperflow.load(hyperflow.model_path(unet_name)))


class HyperFlowSigmas(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionHyperFlowSigmas",
            display_name="Vision HyperFlow Sigmas",
            category=CATEGORY,
            description="The step schedule of a HyperFlow checkpoint: 4 steps for speed, 6 to balance, 8 for quality.",
            inputs=[io.Model.Input("model"), io.Int.Input("steps", default=6, min=4, max=8, step=2, optional=True)],
            outputs=[io.Sigmas.Output(display_name="sigmas")],
        )

    @classmethod
    def execute(cls, model, steps=6) -> io.NodeOutput:
        return io.NodeOutput(hyperflow.sigmas(model, 6 if steps is None else steps))


class LivePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionLivePreview",
            display_name="Vision Live Preview",
            category=CATEGORY,
            description="Show the whole clip moving, with sound, at every sampling step. Does nothing without taeh3 in models/vae_approx.",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("audio_vae", optional=True, tooltip="H3 audio VAE. With it the preview has sound."),
                io.String.Input("folder", default="minimax_flow/preview", tooltip="Where preview clips go, inside the output folder."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, folder, audio_vae=None) -> io.NodeOutput:
        return io.NodeOutput(live_preview.attach(model, folder, audio_vae))


def _blur(mask, radius):
    """A soft edge: a separable box blur, twice (close to a Gaussian), on a [H, W] mask."""
    radius = int(radius)
    if radius < 1:
        return mask
    k = 2 * radius + 1
    m = mask[None, None]
    kernel_h = torch.ones(1, 1, 1, k, dtype=m.dtype, device=m.device) / k
    kernel_v = torch.ones(1, 1, k, 1, dtype=m.dtype, device=m.device) / k
    for _ in range(2):
        m = torch.nn.functional.conv2d(torch.nn.functional.pad(m, (radius, radius, 0, 0), mode="replicate"), kernel_h)
        m = torch.nn.functional.conv2d(torch.nn.functional.pad(m, (0, 0, radius, radius), mode="replicate"), kernel_v)
    return m[0, 0]


def _grow(mask, pixels):
    """The mask made bigger by about this many pixels (a max filter), on a [H, W] mask."""
    pixels = int(pixels)
    if pixels < 1:
        return mask
    k = 2 * pixels + 1
    return torch.nn.functional.max_pool2d(mask[None, None], k, stride=1, padding=pixels)[0, 0]


class LoadMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionLoadMask",
            display_name="Vision Load Mask",
            category=CATEGORY,
            description=("Read a mask the app painted (white = change) from the output folder, sized to width x height, "
                         "grown and with a soft edge."),
            inputs=[
                io.String.Input("file", tooltip="Path inside the output folder."),
                io.Int.Input("width", default=1024, min=16, max=16384),
                io.Int.Input("height", default=1024, min=16, max=16384),
                io.Int.Input("grow", default=8, min=0, max=256, tooltip="Pixels to make the mask bigger by."),
                io.Int.Input("feather", default=12, min=0, max=256, tooltip="Pixels of soft edge."),
            ],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, file, width, height, grow, feather) -> io.NodeOutput:
        with Image.open(output_file(file)) as image:
            image = image.convert("RGBA")
            # Painted strokes: white, or anything not see-through on a clear background.
            alpha = np.array(image.getchannel("A")).astype(np.float32) / 255.0
            light = np.array(image.convert("L")).astype(np.float32) / 255.0
            mask = np.maximum(light * alpha, 0) if alpha.min() < 1 else light
            mask = Image.fromarray((mask * 255).astype(np.uint8)).resize((int(width), int(height)), Image.BILINEAR)
        m = (torch.from_numpy(np.array(mask).astype(np.float32) / 255.0) > 0.5).float()
        m = _blur(_grow(m, grow), feather).clamp(0, 1)
        return io.NodeOutput(m[None])


class CropToMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionCropToMask",
            display_name="Vision Crop To Mask",
            category=CATEGORY,
            description=("Cut out the part of the picture around the mask, with some room around it, scaled so its long "
                         "side is size (in steps of 16): for re-rendering a face at full detail, then Vision Paste Crop."),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("padding", default=0.35, min=0.0, max=2.0, step=0.05, tooltip="Room around the mask, as a share of its size."),
                io.Int.Input("size", default=1024, min=256, max=4096, step=16),
            ],
            outputs=[io.Image.Output(display_name="crop"), io.Mask.Output(display_name="crop_mask"),
                     io.String.Output(display_name="box")],
        )

    @classmethod
    def execute(cls, image, mask, padding, size) -> io.NodeOutput:
        _, H, W, _ = image.shape
        m = mask[0] if mask.dim() == 3 else mask
        if m.shape[-2:] != (H, W):
            m = torch.nn.functional.interpolate(m[None, None], size=(H, W), mode="bilinear")[0, 0]
        ys, xs = torch.nonzero(m > 0.05, as_tuple=True)
        if len(ys) == 0:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            side = max(x1 - x0, y1 - y0) * (1 + padding)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            side = min(side, W, H)
            x0 = int(round(min(max(cx - side / 2, 0), W - side))); y0 = int(round(min(max(cy - side / 2, 0), H - side)))
            x1, y1 = x0 + int(round(side)), y0 + int(round(side))
        w, h = x1 - x0, y1 - y0
        k = size / max(w, h)
        tw, th = max(16, int(round(w * k / 16)) * 16), max(16, int(round(h * k / 16)) * 16)
        crop = image[:, y0:y1, x0:x1, :].movedim(-1, 1)
        crop = torch.nn.functional.interpolate(crop, size=(th, tw), mode="bicubic", antialias=True).clamp(0, 1).movedim(1, -1)
        cm = torch.nn.functional.interpolate(m[y0:y1, x0:x1][None, None], size=(th, tw), mode="bilinear")[0]
        return io.NodeOutput(crop, cm, "%d,%d,%d,%d" % (x0, y0, w, h))


class PasteCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMVisionPasteCrop",
            display_name="Vision Paste Crop",
            category=CATEGORY,
            description="Put a re-rendered crop from Vision Crop To Mask back into the picture, blended by its mask.",
            inputs=[io.Image.Input("image"), io.Image.Input("crop"), io.Mask.Input("crop_mask"), io.String.Input("box")],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, image, crop, crop_mask, box) -> io.NodeOutput:
        x0, y0, w, h = (int(v) for v in box.split(","))
        out = image.clone()
        piece = torch.nn.functional.interpolate(crop.movedim(-1, 1), size=(h, w), mode="bicubic", antialias=True).clamp(0, 1).movedim(1, -1)
        m = crop_mask if crop_mask.dim() == 3 else crop_mask[None]
        m = torch.nn.functional.interpolate(m[:, None], size=(h, w), mode="bilinear")[:, 0, :, :, None].to(out.dtype)
        region = out[:, y0:y0 + h, x0:x0 + w, :]
        out[:, y0:y0 + h, x0:x0 + w, :] = region * (1 - m) + piece.to(out.dtype) * m
        return io.NodeOutput(out)


NODE_CLASS_MAPPINGS = {"MMVisionLoadStill": LoadStill, "MMVisionLoadVideo": LoadVideo, "MMVisionLoadGuide": LoadGuide, "MMVisionMarkPeople": MarkPeople, "MMVisionLora": Lora, "MMVisionHyperFlowLoader": HyperFlowLoader,
                       "MMVisionHyperFlowSigmas": HyperFlowSigmas, "MMVisionLivePreview": LivePreview,
                       "MMVisionLoadMask": LoadMask, "MMVisionCropToMask": CropToMask, "MMVisionPasteCrop": PasteCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"MMVisionLoadStill": "Vision Load Still", "MMVisionLoadVideo": "Vision Load Video", "MMVisionLoadGuide": "Vision Load Guide Video", "MMVisionMarkPeople": "Vision Mark People", "MMVisionLora": "Vision H3 LoRA",
                              "MMVisionHyperFlowLoader": "Vision HyperFlow Loader",
                              "MMVisionHyperFlowSigmas": "Vision HyperFlow Sigmas", "MMVisionLivePreview": "Vision Live Preview",
                              "MMVisionLoadMask": "Vision Load Mask", "MMVisionCropToMask": "Vision Crop To Mask",
                              "MMVisionPasteCrop": "Vision Paste Crop"}
