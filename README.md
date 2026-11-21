# MiniMax H3 Vision

A simple prompt-and-gallery studio for MiniMax H3 videos and MOSS-Image pictures, inside ComfyUI.

Open it from the **MiniMax H3 Vision** button in ComfyUI's sidebar (or View → Open MiniMax H3 Vision).
It also opens by itself the first time ComfyUI loads in a browser tab (a reload in the same tab doesn't
open it again); its **×** goes back to ComfyUI. Switch that off in ComfyUI's settings under
**MiniMax H3 Vision → Open MiniMax H3 Vision when ComfyUI opens**.
It opens full screen over ComfyUI; the gear in the top-right corner has **Exit MiniMax H3 Vision**, which brings you back.

## Install

In ComfyUI's `custom_nodes` folder:

```bash
git clone https://github.com/Timster666/comfyui-minimax-h3-vision
```

then restart ComfyUI. To update later, run `git pull` in that folder (and restart ComfyUI when
Python files changed). It needs nothing else installed: no other node pack, and no extra Python
packages beyond ComfyUI's own.

The repository used to be called `comfyui-minimax-flow`; that address still leads here, and a copy
cloned under the old name keeps working.

## Video, Image, Klein 9b, Pruna and Edit

The app has five tabs, and it always opens on **Video**.

- **Video** makes clips with MiniMax H3. Reference pictures go in the prompt bar, in order: **Picture 1**,
  **Picture 2** … (at most nine). The prompt is sent to H3 as written, so with pictures you write it the way
  MiniMax does, defining the subjects yourself:
  `<Subject 1> is a woman in <Picture 1> and <Picture 2>. <Subject 1> walks into the café.`
  Type `@` to pick a picture, click its label, or drag it onto the text to write its `<Picture N>`; drag a
  picture onto another to change the order. A `<Picture N>` with no picture N is flagged, and isn't sent.
  Older clips (with `@names`) still render as they did.

**RefMods** (optional, with Luisa's [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)
node pack; on RunPod `ENABLE_REFMOD=true`): a RefMod is a reference saved as a small file, made from pictures with
H3's video VAE in a minute, without training. In the Video settings, **RefMods** lists the ones a render carries (up
to eight, each with a strength), **+ Add a RefMod** picks one, and **+ Make one** makes one from the prompt bar's
pictures: a name, what it holds (a person, clothing, a style, a place), Compressed (quick to render with) or Full
(more detail, renders several times slower), and an optional description. When it's made it's added to the
settings. A RefMod has no tag: describe who is in the shot in the prompt. Without the pack, Vision renders as before. With Google
Drive set up, RefMods are kept in a `refmods` folder in the Drive folder too: the ones a pod lacks are fetched the
first time the Video settings look at RefMods after ComfyUI starts, a new one is uploaded when it's made, and
**Sync now** copies both ways. Nothing is deleted on either side.

**Realism** (Video settings, on at 1.2 when `MMH3-Realism-V2.safetensors` is in `models/loras`, V1 as a fallback):
the MMH3 Realism Slider LoRA on every video, as the H3 editor ran it on finals: it sits out the steps above sigma
0.75, which set the composition, and joins for the detail, for natural, un-airbrushed skin. A clip keeps its
strength, so Refine faces and **Use this prompt and settings** use it too.

**Swap character** (with Alissonerdx's [Character Swap v1.1 and Head Swap v1.0](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI)
LoRAs, drbaph's DMD 8-step turbo LoRA and MiniMax H3's own ref2va model; BiRefNet optional; on RunPod `ENABLE_SWAP=true`):
a finished or imported video's ⋯ menu › **Swap character…** puts the person from a picture into the video, keeping its
motion, lips, camera and sound (first 15 s, at 0.4–1.0 MP in the video's shape). **Character** replaces the whole
person, **Head only** (experimental) just the head. The prompt is the author's template; its one gap, the character
(or the face) in words, is typed in the dialog or, left empty, written from the picture by MOSS-Image's text model
while it renders. It follows the published native-node workflow: the video as an `MiniMaxH3AddGuide` guide at frame 0,
the picture as a reference on grey (at up to 1 MP, H3's largest reference size, as Henge sends them), 8 steps, Euler (Beta for heads), CFG 1. The templates are Henge's (MIT). With several
people in the shot, **Who to replace › Pick on the first frame** (Character, with SAM 3.1 in `models/checkpoints`)
lets you click the one to replace: SAM 3 follows them through the clip and paints them red (solid, or see-through so
the face shows), the prompt says to replace only the red-marked person, and **Check the mark** shows the painted
video first. That is how Henge picks a person; it works best in one continuous shot. **Model** is the author's (MiniMax H3's own
model) by default; any other installed H3 model but HyperFlow can be tried (untested): a turbo build (H3 Turbo,
10Eros, PRISM, DaSiWa turbo) runs with its own steps and sampler without the DMD LoRA, a full one (Singularity,
DaSiWa) gets the DMD LoRA and 8 steps.

**Motion from** (Video tab, with Alissonerdx's Motion Transfer v1.0 LoRA; on RunPod `ENABLE_SWAP=true`): next to
Add image, **Motion from…** then a click on a video in the gallery (or a video's ⋯ menu › **Use its motion**) makes
the next render take that video's motion, timing, camera and sound (first 15 s, in its shape). The prompt bar's
pictures and prompt say who and what performs it: up to 9 pictures, each a person, another view of them, an outfit,
an object, a place or a style, and the prompt says which is which. **Use the author's prompt** puts his six-part
`motion_transfer:` template in the prompt to edit. As in his native workflow: the video as a guide at frame 0,
pictures at up to 1 MP each at H3's largest reference size, DMD 8-step turbo, Euler, CFG 1 on MiniMax H3's own model;
another model in the Video settings is untested (turbo models run their own way). Your LoRAs are added; RefMods and
Realism aren't. The chip's menu picks the method: **TARA · several people** (default when installed: the author's Motion
Transfer TARA v1.0 LoRA, trained for several people, objects or vehicles, run as in his native TARA workflow — the
video as a native `<Video 1>` reference instead of a traced guide, with his temporal alignment patch from
[ComfyUI-BFSNodes](https://github.com/alisson-anjos/ComfyUI-BFSNodes) at the training defaults; 2 to 15 s of video)
or **Motion Transfer v1.0** (the traced guide, his default for one performer). **Pick the people** on the chip: click each person
on the video's first frame (up to 4) and choose which picture they become; SAM 3.1 follows each through the clip and
paints them see-through in their colour (red, blue, green, yellow), and with the prompt left empty Vision writes the
author's prompt for them (Henge's one- or several-person recipe with `motion_transfer:`), each picture described by
MOSS-Image's text model; the people in the video are never described, so their looks don't leak. **Check the marks**
shows the painted video first. The author recommends TARA for marked people.

**Refine faces** (optional, with Carasibana's [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine),
the `MiniMaxH3NativeAudioLock` node and the `face_yolov8m.pt` detector in `models/ultralytics/bbox`; on RunPod
`ENABLE_FACEREFINE=true`): a finished video's ⋯ menu › **Refine faces…** tracks the face on every frame, re-renders it
up close with H3 at a low denoise (Light, Normal or Strong) using the clip's prompt, pictures, LoRAs and RefMods, with
its own sound locked in for the lips, and stitches it back. An optional face picture from Add image joins the
references and picks which face to follow; without one, the largest face is refined. The result is a new clip with
the original sound. HyperFlow can't refine (its schedule is fixed), so those clips are refined with another model.

The **eye** in the prompt bar (every tab) shows exactly what the model would get, if you rendered now: the
prompt text after Vision's changes, the negative prompt or the upsampler's request where there is one, and the
pictures in the order they're sent. Point at it to see it, click it to keep it open; **Copy** copies the text.
- **Image** makes pictures with MOSS-Image. It's good at words in pictures: put the text you want
  in quotes. A finished image can be sent to the Video tab with **Use in a video**, and the Video
  tab's **Add image** pop-up lists all of them under **Generated images**.
  **Edit** (next to Negative prompt) edits one picture instead: a slot for the picture appears in the
  prompt bar (click it, or drag one in from the grid, Add image or your computer), and the prompt
  describes only the change ("Change only the pink blazer to a black wool coat. Keep everything else
  unchanged."). The new image keeps the picture's shape, at the chosen resolution.
- **Klein 9b** edits pictures with FLUX.2 [klein] 9B. The prompt bar has four slots, **image 1** to
  **image 4**, and the prompt calls the pictures by those names ("put the bag from image 2 in the hand
  of the woman in image 1"). The slots keep their names: drag a picture onto another slot to swap
  them, or drop a new one on a slot to replace it, and the prompt stays as it is. Fill a slot by
  clicking it (it opens **Add image**), or drag a picture onto it from the grid, Add image or your
  computer. Under each slot is the picture's own size, and hovering it shows it big. Drag a slot onto the
  prompt text to put its name ("image 2") where the cursor was. Click a filled slot to open **Add image**
  and pick a picture to replace it. With no pictures
  Klein makes one from the prompt alone. **Size** in the settings is **Match image 1** (its shape) or an
  aspect ratio. **Edit this** on a result puts it in image 1, to keep editing; the Image tab's pictures
  have **Edit in Klein**. A prompt that names an empty slot isn't sent.

  **Face swap** (the switch at the top of Klein's prompt bar) puts a head on another picture with
  Alissonerdx's [BFS head-swap LoRA](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) for Klein 9B
  (rank 128 or rank 64, picked in the settings). **body** is the picture that keeps its pose, light and
  background, **face** the person whose head goes onto it, and up to three more face pictures (other
  angles) help with who it is. The app writes the swap prompt (the "faceswap detailer" workflow's); the
  text box adds notes after it. Everything goes in at the swap's size, 2 MP to start with.
  - **Paint head** (under the body) and **Only the head**: paint the head and neck, and only that part
    changes; the rest of the body picture stays exactly as it was. Its sampler is the standard one or
    [LanPaint](https://github.com/scraed/LanPaint) (better blending, slower), and the mask can be grown
    and given a soft edge in the settings.
  - **Refine face** (with Only the head): a second pass that cuts the head out at 1024 px, renders it
    again and pastes it back, for sharper detail.
  - **Before / after** on a swap's page: drag across the picture to compare it with the body picture.

  It needs `bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors` or `..._step3750_rank64.safetensors`
  in `models/loras` (any folder). Only use it on people who agreed to it.
- **Pruna** makes and edits pictures with Pruna AI's P-H3-Image. It has two modes, switched at the top of
  the prompt bar:
  - **Image**: up to nine slots, **picture 1** to **picture 9** (more appear as you fill them). With no
    pictures it makes one from the prompt; with pictures, picture 1 is the one edited and the others are
    references ("put the lamp from picture 2 on the table in picture 1"). The slots work like Klein's.
  - **Character sheet**: the character's picture in the first slot, and up to three more that help with
    its **Face**, **Outfit** or **Details**. Pick the views (head and full body from the front, ¾, side and
    back, and close-ups of the hands and feet) or a preset. The app writes the sheet's prompt; the text
    box is for notes. The sheet's shape follows the views, at 8 MP.

  **Enhance prompt** (in the bar, off by default) runs Pruna's prompt upsampler first: it rewrites the
  text into a long, detailed prompt. **What the model gets** shows the exact text that is sent, and a
  picture's page shows what the upsampler wrote, with **Use the upsampled prompt**. Sampling is always
  8 steps. Every image has **Edit in Pruna**.
- **LoRAs** are added in the settings pop-up with **+ Add LoRA**: a window lists the LoRA files to
  search and pick from. **Refresh** looks in the folders again, so LoRAs you download while ComfyUI runs
  show up without a restart. Sort your LoRAs by model in folders inside `models/loras`:

  ```
  models/loras/
    MiniMax H3/    the Video tab's LoRAs
    MOSS Image/    the Image tab's LoRAs
    Flux Klein/    the Klein 9b tab's LoRAs
    P-H3 Image/    the Pruna tab's LoRAs (it lists MiniMax H3's too: P-H3-Image runs on MiniMax H3)
  ```

  Each tab lists its own folder first, then **Not sorted**: files loose in `loras` and any folder that
  isn't one of these. Another model's folder is hidden unless you tick **Show all LoRAs**. Folder names
  are matched loosely (`minimax_h3`, `moss`, `klein` work too). Moving a LoRA into a folder doesn't break
  saved settings or **Use this prompt and settings** on older clips: a LoRA is found by its file name.

  New LoRAs can be added from the same window, into the folder picked under **New LoRAs go into**
  (this tab's folder at first):
  - **Upload** (or drop `.safetensors` files on the window) sends a file from your computer in 8 MB
    pieces. A piece that fails is sent again, so a dropped connection slows an upload down rather than
    losing it. The file is checked to be a safetensors file, and a LoRA with the same name is only
    replaced if you say so.
  - **Paste a link** and click **Get**: the pod downloads it, so it's fast and goes on with the window
    closed. Hugging Face links are to the file (open it on the site and copy the address); Civitai
    links are to the model page (with `?modelVersionId=` for a particular version, otherwise the
    newest) or an `api/download` link; any other https link to a `.safetensors` file works too.
    Gated Hugging Face repositories need `HF_TOKEN` on the pod, and Civitai LoRAs that need a login need
    `CIVITAI_TOKEN`.

  Partial files are kept in `models/.downloads/loras`, so ComfyUI never lists half a LoRA.
- **Edit** puts your videos together, like a very simple CapCut: your videos on the left, a player
  on the right and one timeline track below. Add a video with its **+**, by double-clicking it or
  with Enter (it goes into the gap between clips nearest the playhead), or drag it onto the timeline
  (it goes into the gap nearest where you drop it); drag clips to move them and their edges to trim them. **S**
  splits at the playhead, **⌫** deletes, **⌘Z** undoes, Space plays. **Export** saves one MP4
  (H.264 and AAC, 24 fps) in `minimax_flow/exports/`, at the videos' own size, 720p or 1080p; videos
  of another shape get black bars. It downloads by itself when done. Quality is visually lossless:
  CRF 16, and clips already at the export size are passed to the encoder without any conversion.
  The edit is saved as you go.
  **Project → Export project** saves the timeline, the videos it uses and their reference images
  (the `@images` in their prompts) as one `.zip` (in
  `minimax_flow/exports/`, and it downloads). **Import project** opens such a zip on any machine: its
  videos join your Video gallery and the Edit tab, its reference images join **Add image**, and its
  timeline replaces the current one (it asks first). If an image's name is taken, it gets a number (`detective2`) and
  the videos' prompts are changed to match, so **Use this prompt and settings** finds the right picture. Videos
  that are already there aren't copied twice.

It needs no other custom node pack. Videos run on ComfyUI's own MiniMax H3 nodes plus a few of
this pack's own (in the node menu under **MiniMax H3 Vision**), and images on ComfyUI's own nodes.
A tab whose models aren't installed shows a **Not installed** badge; click it to see which files
are missing.

While a clip renders you see its steps and a moving live preview (with `taeh3`, below). While an
image renders you see its steps and ComfyUI's live preview. MOSS-Image follows Comfy-Org's template
(er_sde / simple, the recommended negative prompt), at 50 steps and guidance 3.0.

### Video sizes

The Video tab's **Resolution** slider goes from 0.1 to 2 MP; click the size to type any value in between (0.98,
say). Sides are multiples of 32, as MiniMax asks. **1.0 MP is H3-Base's own size**, as large as its
768 × 1344 pixel limit allows in each shape: 1344×768 (16:9), 768×1344 (9:16), 992×992 (1:1), 1152×864 (4:3),
864×1152 (3:4) and 1536×672 (21:9). Sizes up to 1 MP stay within that limit, small sizes are picked to keep
the frame's shape, and 1.5 and 2 MP go above H3's own size on purpose.

## MiniMax H3 files

The models the Video tab needs, into ComfyUI's `models` folder:

| File | Folder | Where to get it |
|---|---|---|
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), `text_encoders/` |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | Comfy-Org/MiniMax-H3, `vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | Comfy-Org/MiniMax-H3, `vae/` |

and at least one model (each model in the Video tab's list is one of these families):

| Model | Files |
|---|---|
| HyperFlow H3 Flash | `minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors` and `minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors` in `models/diffusion_models/hyperflow/`, from [VideoRebirthAI/hyperflow-h3-flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash). Its GGUF files don't load here. |
| MiniMax H3 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` and `minimax_h3_fl2va_pruned_int8_convrot.safetensors` in `models/diffusion_models/` |
| MiniMax H3 Turbo | the same two, and `minimax_h3_turbo_v4_step600_ema.safetensors` in `models/loras/` |
| Singularity v2, DaSiWa, 10Eros, PRISM | their checkpoint in `models/diffusion_models/` (names in `h3_graph.py`) |

The text encoder and video VAE in other precisions (`..._int8_convrot`, `..._bf16`) work too.
Optional: `taeh3.safetensors` in `models/vae_approx/`, from [OzzyGT/taeh3](https://huggingface.co/OzzyGT/taeh3),
for the moving live preview.

## MOSS-Image files

From [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image), into ComfyUI's `models` folder:

| File | Folder |
|---|---|
| `moss_image_sft_int8_convrot.safetensors` | `models/diffusion_models/` |
| `qwen3.5_9b_w4a8_gptq.safetensors` | `models/text_encoders/` |
| `flux2-vae_small.safetensors` | `models/vae/` |

Any other MOSS-Image checkpoint in `models/diffusion_models` (BF16, Base, a finetune) shows up
in the Image tab's model list as well.

**Turbo (optional).** [Spectrallic's 8-step DMD checkpoint](https://huggingface.co/Spectrallic/MOSS-Image-SFT-8step-DMD-int8-convrot)
(`moss_image_sft_8stepdmd_int8_convrot.safetensors`, 5.4 GB) also goes in `models/diffusion_models/`
and uses the same text encoder and VAE. It shows up as "MOSS-Image SFT Turbo 8-step (INT8)" and
renders with 8 steps, guidance 1, Euler and no negative prompt. Any checkpoint whose file name
has "dmd", "turbo" or "8step" gets these turbo settings.

**Edit (optional).** Comfy-Org's edit LoRA, [`moss_image_edit_lora_rank_64_bf16.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/loras/moss_image_edit_lora_rank_64_bf16.safetensors)
(297 MB), goes in `models/loras/` (best in `MOSS Image/`). Edits follow ComfyUI's MOSS-Image edit
workflow, without its Skip Layer Guidance: the LoRA at strength 1, the instruction and the
negative encoded together with the picture (`TextEncodeMOSSImageEdit`, format Moss-Image-Edit), 30
steps, guidance 4, euler/normal. The picture is scaled to the chosen resolution in steps of 16 px, and
the new image is made at that size. It needs ComfyUI 0.43.0 or newer.

## Klein 9B files

From Black Forest Labs (accept the FLUX Non-Commercial License on each repository first; they need a
Hugging Face login) and [Comfy-Org/flux2-klein-9B](https://huggingface.co/Comfy-Org/flux2-klein-9B):

| File | Folder |
|---|---|
| `flux-2-klein-9b.safetensors` ([FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)), or `flux-2-klein-9b-fp8.safetensors` ([FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8)) | `models/diffusion_models/` |
| `qwen_3_8b_fp8mixed.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` | `models/vae/` |

Distilled checkpoints render in 4 steps without guidance. `flux-2-klein-base-9b(-fp8).safetensors`
(the undistilled Base) shows up too, with 20 steps and guidance 5. The graph follows ComfyUI's Klein 9B
image edit templates: each picture is scaled to 1 MP and passed to the model with `ReferenceLatent`.
Klein numbers the pictures in the order it gets them, so when a slot before it is empty the prompt the
model reads is renumbered to match (slots 1 and 3 filled: "image 3" is sent as "image 2"). MOSS-Image's
`flux2-vae_small` isn't used for Klein: its pictures need the full VAE encoder.

## Prompt builder

**Builder** in the prompt bar opens a library of your own prompts, in three panels:

- **Categories** and their subcategories: **+ Category**, and on each row (when you point at it) add a
  subcategory, rename (or double-click the name) and delete. Drag rows to reorder them, or a subcategory onto
  another category to move it there.
- **Prompts** of the category picked on the left (or, with the search field, of all of them). Click a prompt to
  add it to the text field, at the cursor, with a space after it. **Add prompt** at the top and bottom of the list;
  on each prompt Edit, Move (to any category), Delete (with Undo) and Add (a new one below it). Drag a prompt onto
  another prompt to put it before that one, or onto a category or subcategory to move it there. A prompt has an
  optional name and its text.
- **Prompt**: the text field, starting with what's in the prompt. **Reset** goes back to that, **Cancel** closes
  without changing the prompt, **Apply** (Ctrl/⌘+Enter) puts the text into the prompt. **Save as prompt** keeps the
  text in the library, in the category picked on the left.

Changes to the library are saved at once, in `minimax_flow/prompt_library.json`. With Google Drive set up, the
library is also kept in `vision-prompts.json` in the Drive folder: fetched when the builder is first opened after
ComfyUI starts, and saved there a few seconds after each change. Two pods' changes are merged prompt by prompt (the
newer change wins, and deletes are kept), so the library follows you to every new pod.

## Google Drive

Edit projects and video `.zip`s can be saved to a Google Drive folder and opened from it: **Edit › Project ›
Save project to Google Drive** and **Open from Google Drive**, a video's **⋯ › Save .zip to Google Drive**, and
**Google Drive** in the gear menu. The machine ComfyUI runs on copies straight to and from Drive with
[rclone](https://rclone.org), one file at a time, with progress and Cancel; the window can be closed meanwhile.
A name that's taken in the folder gets `_2`, `_3`…

It needs `rclone` on the machine and two environment variables: `GDRIVE_FOLDER`, the folder's link, and
`GDRIVE_TOKEN`, the line in `{ }` that `rclone authorize "drive"` prints after you sign in to Google (run it on any
computer with a browser). Without them, nothing about Drive is shown. The RunPod template installs rclone; its
README has the steps.

## Models page

**Gear menu → Models** lists the model files each tab uses (`model_catalog.json`), in groups, with what is
installed, sizes and the free disk space. **Download** fetches a group from Hugging Face in the background on
the machine ComfyUI runs on (one file at a time, into the right `models/` folder); a model shows up in its tab
when it is in. Gated models (Klein) need `HF_TOKEN` set where ComfyUI runs, and the license accepted on Hugging
Face. **Delete** removes a group's files, keeping any another installed group needs. A tab's **Not installed**
opens the page at that tab's models.


Gated models (Klein) need a Hugging Face read token: the pod's `HF_TOKEN`, or paste one at the top of the Models
page. A pasted token is checked with Hugging Face and used until ComfyUI restarts; with **Remember on this pod** it
is kept in `models/.downloads/hf_token` (readable only by its owner) until you click **Remove**. It is also used for
LoRA links to gated Hugging Face repositories.

## P-H3-Image files (Pruna tab)

ComfyUI 0.44 or newer (it has P-H3-Image's nodes built in). From
[PrunaAI/P-H3-Image-Comfy](https://huggingface.co/PrunaAI/P-H3-Image-Comfy), plus MiniMax H3's FL2VA model:

| File | Folder |
|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` ([Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)) | `models/diffusion_models/` |
| `p-h3-image_lora_bf16.safetensors` | `models/loras/` (any folder) |
| `p-h3-image_text_encoder_int8_convrot.safetensors` | `models/text_encoders/` |
| `p-h3-image_vae_int8_convrot.safetensors` | `models/vae/` |
| `p-h3-image_prompt_upsampler_bf16.safetensors` (for Enhance prompt; the INT8 or W4A8 one works too) | `models/text_encoders/` |

The graph follows ComfyUI's P-H3-Image templates: the LoRA at 1.0 on FL2VA, ModelSamplingMiniMaxH3 at 1.0,
"p-h3-image Conditioning" with the pictures, and a fixed 8-step schedule (er_sde, guidance 1). If Pruna's
full checkpoint (`p-h3-image_int8_convrot.safetensors` or another) is in `models/diffusion_models`, the
settings offer it as a second model. The model calls the pictures `<Picture 1>`, `<Picture 2>` … in the
order it gets them, so "picture 3" with slot 2 empty is sent as `<Picture 2>`.

### MiniMax H3-Image (in the Pruna tab)

MiniMax's own image model is offered as another model in the Pruna tab, to compare with P-H3-Image on the
same prompt and pictures. It needs ComfyUI 0.45 or newer. From
[Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image), plus MiniMax H3's text encoder
(`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, or its INT8 or BF16 version):

| File | Folder |
|---|---|
| `minimax_h3_image_pruned_int8_convrot.safetensors` (or `_w4a8`, smaller, or `_bf16`) | `models/diffusion_models/` |
| `minimax_h3_image_vae_fp16.safetensors` | `models/vae/` |

The graph follows ComfyUI's H3-Image templates: "MiniMax H3 Image Conditioning" with the pictures, 6 steps,
euler/simple at CFG 1, the prompt zeroed out as the negative. It reads the prompt as written (no prompt
expander, no Enhance prompt), and MiniMax H3's LoRAs don't fit it. Sizes are 4, 8, 12 and 16 MP: MiniMax's
sweet spot is about 16 MP (5472×3072 at 16:9), and below 4 MP it looks soft.

## File names and importing

Finished files are named after the start of their prompt, like
`vision_a_detective_in_a_long_coat_walks_00001_.mp4`, and downloads keep that name.

Each file also carries its prompt and settings inside it (ComfyUI's metadata, under `mmflow`).
**Import**, above the grid, brings a video or image back into the grid with
them, ready for **Use this prompt and settings**. You can also drop files on the page: videos
anywhere, images in the Image tab. Older files without `mmflow` work too when ComfyUI saved its
workflow in them: clips from earlier versions of this app or from the H3 editor, and MOSS-Image
images. Their settings are worked out from that workflow. Files saved with ComfyUI's
`--disable-metadata` have nothing to read.

**Delete**, to the right of Import, deletes several videos or images at once. Tick them with the
box in a tile's top-right corner, or ⌘-click (Ctrl-click on Windows) to add or take out one and
Shift-click to tick a range. While anything is ticked, a plain click ticks too; Esc or **Clear
selection** lets go. Delete asks before it deletes, and the files go from the output folder too.

To move a video together with its reference images, open it and click **Export .zip** (or
right-click it). The `.zip` holds the video and the pictures its `@images` name, and downloads. Import it
with **Import** or by dropping it on the page, here or on another machine: the video joins the grid and
its pictures join **Add image**, renamed with a number if the name is taken, just like a project. The
Edit tab's **Import project** takes these zips too, and adds the video to the media panel.

## Where things are kept

In ComfyUI's output folder, under `minimax_flow/`:

- `project.json`: your images, clips and last-used settings
- `images/`: uploaded reference images
- `clips/`: finished videos
- `stills/`: finished images
- `exports/`: exported edits
- `strips/`: the timeline's filmstrips and waveforms (made again when deleted)
- `preview/`: live previews while rendering (cleaned up as it goes)

## Files

- `__init__.py`: tells ComfyUI about the plugin and its nodes
- `flow_server.py`: the plugin's server: its API and storage
- `h3_graph.py`: video graphs: the MiniMax H3 model families and the prompt H3 expects
- `vision_nodes.py`: the pack's graph nodes, with `h3_lora.py` (LoRAs), `hyperflow.py` (HyperFlow H3 Flash)
  and `live_preview.py` (the moving preview)
- `moss_bridge.py`: image graphs, with MOSS-Image
- `klein_bridge.py`: Klein 9B graphs, with up to four reference pictures
- `ph3_bridge.py`: P-H3-Image graphs (Pruna tab), with up to nine pictures and the prompt upsampler
- `model_store.py` and `model_catalog.json`: the Models page's list and background downloads
- `prompt_lib.py`: the prompt builder's library, and keeping it in step with Google Drive
- `app/builder.js`: the prompt builder window
- `refmod_bridge.py`: RefMods (ComfyUI-MiniMaxH3Mod): making them, and adding them to a render
- `gdrive.py`: saving project zips to, and opening them from, a Google Drive folder with rclone
- `lora_in.py`: new LoRAs from the LoRA window: uploads in pieces, and Hugging Face and Civitai links
- `edit_media.py`: the Edit tab's filmstrips, waveforms and export, with PyAV
- `edit_project.py`: edit projects as zip files (export and import)
- `clip_info.py`: the prompt and settings saved in each file, and reading them back on import
- `comfy_link.py`: sending jobs to ComfyUI's queue, following and stopping them
- `web/minimax_flow_launcher.js`: the sidebar button, and opening on start
- `app/index.html`, `app/app.js`: the app itself, and `app/edit.js`: its Edit tab

## License

GPL-3.0 (`LICENSE`). The video code is adapted from the
[H3 Editor](https://github.com/igorls/comfyui-h3-editor) by Igor Lins e Silva, which includes work
from ComfyUI-HyperFlow-Flash and ComfyUI-MiniMax-H3-Turbo. `NOTICE` lists what came from where.

The models have their own licenses: MiniMax H3's community license, MOSS-Image's, and the FLUX
Non-Commercial License for FLUX.2 [klein] 9B (non-commercial use; commercial use goes through Black
Forest Labs' API).
