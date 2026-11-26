# MiniMax H3 Vision: the API, for other front ends

Vision's page talks to its server through the routes below. Another UI can use the same routes, from a page served
by the same ComfyUI (same origin, so no CORS and no login). This file covers what a simpler UI needs. Everything else
Vision does (Edit, Klein, MOSS, Pruna, Swap, RefMods, Drive, the model downloader) is in `flow_server.py`'s `ROUTES`,
and each handler's docstring says what it takes.

**Keeping in step.** These routes were made for Vision's own page, not as a frozen contract. When a change in Vision
alters anything described here, the same change updates this file. Check its history (`git log -p API.md`) after
pulling Vision.

Errors are plain text with a 4xx status, written to be shown to the person as they are
(`400 Write a prompt first.`). Success is JSON.

## Shared data

Everything lives in one project, `output/minimax_flow/project.json`, the same one Vision's page shows: a render
started from another UI appears in Vision's gallery, and the other way round. Don't write that file directly; go
through the routes (they lock it).

## Files

Every file is served by ComfyUI's own `/view`:

```
/view?type=output&subfolder=<subfolder>&filename=<file>
```

- A clip: `subfolder` and `file` from its record (videos are in `minimax_flow/clips`).
- A reference (picture or video): `subfolder` = `minimax_flow/images`, `filename` = the reference's `file`.
- A reference video's thumbnail (its first frame, JPEG): `GET /mmflow/api/images/{id}/poster`.

## Start up: `GET /mmflow/api/state`

The whole project and the choices this pod offers. The fields a video UI needs:

```jsonc
{
  "clips": [ /* clip records, newest first: see below */ ],
  "images": [ /* references: see References */ ],
  "options": {
    "root": "minimax_flow",
    "models": [            // installed video models, in menu order
      { "id": "h3_turbo", "name": "MiniMax H3 Turbo", "steps": 8, "minSteps": 1, "maxSteps": 16, "stepSize": 1,
        "sampler": "euler", "scheduler": "simple" }
    ],
    "missing": null,       // when models is empty: why, to show as is
    "aspects": { "16:9": [16, 9], "9:16": [9, 16], "1:1": [1, 1], "4:3": [4, 3], "3:4": [3, 4], "21:9": [21, 9] },
    "videoMP": { "min": 0.1, "max": 2.0 },          // the size range, in megapixels (1 MP = 1024×1024)
    "length": { "min": 2, "max": 20, "fps": 24 },   // seconds
    "loras": ["MiniMax H3/some_lora.safetensors"],  // every LoRA file in models/loras, Klein's too: names as settings take them
    "samplers": [...], "schedulers": [...],
    "realism": { "file": "MMH3-Realism-V2.safetensors", "strength": 1.2, "range": [0, 3], "start_sigma": 0.75 }  // file null: not installed
  }
}
```

The real numbers are in the response; the ones above are examples. HyperFlow (scheduler `hyperflow`) takes 4, 6 or 8
steps and no other scheduler. The project also has `ui` (Vision's own page
state: leave it alone; `PUT /mmflow/api/ui` is Vision's) and `edit` (its editor).

## Make a video: `POST /mmflow/api/generate`

```jsonc
{
  "kind": "video",
  "prompt": "The woman in <Picture 1> walks along the beach at dusk.",
  "pictures": ["i1a2b3c4d5"],   // reference ids, in order (optional: leave out for no references)
  "client_id": "myui-x7f3",     // your WebSocket's clientId, so live previews reach you (optional)
  "settings": {
    "model": "MiniMax H3 Turbo", // a model's name from options.models
    "aspect": "16:9",
    "res": 0.6,                  // megapixels
    "len": 6,                    // seconds
    "steps": 8,                  // within the model's minSteps..maxSteps, in stepSize steps
    "seedMode": "random",        // or "fixed" with "seed": 1234
    "loras": [{ "name": "MiniMax H3/x.safetensors", "strength": 1.0, "on": true }],
    "realism": { "on": true, "strength": 1.2 },   // optional
    "sound": true,               // false: saved without sound
    "sampler": null, "scheduler": null            // optional; null is the model's own
  }
}
```

The reply is `{"clip": {...}}`, the new clip, queued. Its `id` is how you follow it.

**References in the prompt.** With `pictures`, the prompt is sent to H3 as written, and names the references by
kind and order: pictures as `<Picture 1>` … `<Picture 9>`, videos as `<Video 1>` … `<Video 3>`, each kind counted on
its own along the `pictures` list. A tag with no reference behind it is refused with a message. Without `pictures`,
the older `@name` mentions are matched against the references' names.

## Clips

A clip record (the fields you'll use):

```jsonc
{
  "id": "c0123456789", "kind": "video",          // "image" for stills from the image tabs
  "status": "queued",                            // queued → rendering → done, or failed
  "step": 0, "steps": 8,                         // progress: step of steps
  "prompt": "...", "model": "MiniMax H3 Turbo", "seed": 1234,
  "width": 1024, "height": 576, "len": 6, "aspect": "16:9", "res": 0.6,
  "file": null, "subfolder": "minimax_flow/clips",   // file is set when done
  "error": null,                                 // the reason when failed
  "starred": false, "created": 1764170000.0,
  "prompt_id": "...",                            // ComfyUI's job id
  "pictures": [{ "id": "i…", "name": "anna" }, { "id": "i…", "name": "walk", "kind": "video" }],
  "silent": true                                 // only when saved without sound
}
```

- `PATCH /mmflow/api/clips/{id}` `{"starred": true}`: star or unstar.
- `DELETE /mmflow/api/clips/{id}`: delete it and its file; a clip still queued or rendering is cancelled first.

## Following progress: ComfyUI's WebSocket

Open `ws(s)://<host>/ws?clientId=<your id>` (the same id you send as `client_id`). Messages are JSON
`{"type": ..., "data": ...}`:

| type | data | what to do |
|---|---|---|
| `mmflow.clip` | `{"clip": {...}}` | A clip changed (queued, rendering, done, failed): replace your copy by `id`. Sent to everyone. |
| `progress` | `{"prompt_id", "node", "value", "max"}` | ComfyUI's own. When `node` is `"15"` (the sampler) and `prompt_id` is one of your clips', `value` of `max` is its step. |
| `mmvision_preview` | `{"prompt_id", "subfolder", "filename", "step", "total"}` | A short low-res MP4 of the whole video so far, after each step (`/view` with `type=output`). Sent only to the `client_id` the render was queued with. |

A finished clip comes as `mmflow.clip` with `status: "done"` and `file` set. If the socket drops, reconnect and
call `GET /mmflow/api/state` again to catch up.

## References (pictures and videos)

`GET /mmflow/api/state` → `images`:

```jsonc
{ "id": "i1a2b3c4d5", "name": "anna", "file": "i1a2b3c4d5.png", "added": 1764170000.0 }
// a reference video also has:
{ "kind": "video", "seconds": 4.2, "width": 1280, "height": 720,
  "audio": true,     // it has a soundtrack
  "sound": false }   // whether H3 hears it with the video (off unless switched on)
```

- `POST /mmflow/api/images`: multipart, one or more `file` parts (pictures, or videos: any format; a video a browser
  can't play is re-encoded as H.264). Reply: `{"images": [the new references]}`. Names come from the file names, made
  unique.
- `PATCH /mmflow/api/images/{id}` `{"name": "new_name"}`: rename (letters, digits and `_`). `{"sound": true}`: a
  reference video's sound goes to H3 with it.

H3 takes at most 9 pictures and 3 videos in one render. A reference video is used for up to its first 15 seconds and
cut to the render's length.

## Other routes you may want

- `GET /mmflow/api/models`: the model downloader's state (what's installed, downloading, missing).
- `POST /mmflow/api/preview`: the same body as generate; replies with what H3 would get (prompt texts, pictures,
  reference videos, model files) without rendering.
- `GET /mmflow/api/loras`: the LoRA list again (after adding one).
- `POST /mmflow/api/restart`: restart ComfyUI in place; `409 {"busy": n}` while renders are running or queued, unless
  `{"force": true}`.
