"""Edit projects and single videos as zip files, to move or keep.

A project zip holds:
    project.json   {"app": "MiniMax H3 Vision", "kind": "edit-project", "version": 1,
                    "edit": {name, ratio, muted, pieces: [{id, clip, in, out}]},
                    "clips": [the clip records (prompt and settings), each with "zip_file"]}
    clips/<file>   each video the timeline uses, whole, so it can be trimmed differently later
    images/<file>  the reference pictures and videos those videos used, listed in "images" with their
                   names, so the prompts still find them after an import
The videos a video was made from (Motion from…, Swap, Refine faces) go in too, as videos of their own.

A single video's zip is the same with "kind": "clip" and no "edit": one video and its reference pictures.

No ComfyUI imports, so it can be tested anywhere.
"""

import json
import os
import re
import shutil
import uuid
import zipfile

KIND = "edit-project"
CLIP_KIND = "clip"
VERSION = 1
# What a clip keeps when it travels in a project.
CLIP_FIELDS = ("id", "kind", "prompt", "model", "profile", "steps", "cfg", "res", "aspect", "len", "width", "height",
               "seed", "loras", "starred", "created", "imported", "found", "refs", "sampler", "scheduler", "negative",
               "pictures", "refmods", "faces", "realism", "swap", "motion")
IMAGE_FIELDS = ("id", "name", "file", "added", "kind", "seconds", "width", "height", "audio")
# A clip's fields that name the video it was made from.
SOURCE_FIELDS = ("motion", "swap", "faces")
MENTION = re.compile(r"@([A-Za-z0-9_]+)")


class Bad(Exception):
    """Not a zip this app can open, or nothing to put in one; the message says why."""


def refs_of(clip, images):
    """The reference pictures a clip used: the ones it recorded, else the @names in its prompt (any case)."""
    by_id = {i["id"]: i for i in images}
    found = [by_id[r["id"]] for r in clip.get("refs") or [] if r.get("id") in by_id]
    if found:
        return found
    by_name = {i["name"].lower(): i for i in images}
    out = []
    for name in MENTION.findall(clip.get("prompt") or ""):
        image = by_name.get(name.lower())
        if image and image not in out:
            out.append(image)
    return out


def rename_mentions(prompt, names):
    """A prompt with each @old (any case) as @new, for names = {old lower case: new}."""
    return MENTION.sub(lambda m: "@" + names.get(m.group(1).lower(), m.group(1)), prompt or "")


def _unique(name, taken):
    stem, ext = os.path.splitext(name)
    n = 2
    while name in taken:
        name = "%s_%d%s" % (stem, n, ext)
        n += 1
    taken.add(name)
    return name


def sources_of(clip):
    return [clip[k]["from"] for k in SOURCE_FIELDS if isinstance(clip.get(k), dict) and clip[k].get("from")]


def _write(out_path, used, clips, clip_path, images, image_path, head):
    """Write the videos (ids in used), the videos they were made from, their reference pictures and project.json
    to out_path.

    head(ids of the videos that went in) gives project.json's other fields.
    """
    used = list(used)
    for clip_id in used:   # grows as it goes: a source's own source goes in too
        used += [s for s in sources_of(clips[clip_id]) if s in clips and s not in used]
    records, names = [], set()
    temp = out_path + ".part"
    try:
        # Videos are compressed already, so they are stored as they are.
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
            for clip_id in used:
                clip = clips[clip_id]
                path = clip_path(clip)
                if not os.path.exists(path):
                    continue
                name = _unique(os.path.basename(clip["file"]), names)
                z.write(path, "clips/" + name)
                record = {k: clip[k] for k in CLIP_FIELDS if k in clip}
                record["zip_file"] = "clips/" + name
                records.append(record)
            # The reference pictures of the videos that went in.
            pictures, files = [], set()
            for clip_id in [r["id"] for r in records]:
                for image in refs_of(clips[clip_id], list(images)):
                    if image["id"] in {p["id"] for p in pictures} or not image_path or not os.path.exists(image_path(image)):
                        continue
                    name = _unique(os.path.basename(image["file"]), files)
                    z.write(image_path(image), "images/" + name)
                    picture = {k: image[k] for k in IMAGE_FIELDS if k in image}
                    picture["zip_file"] = "images/" + name
                    pictures.append(picture)
            project = {"app": "MiniMax H3 Vision", "version": VERSION}
            project.update(head({r["id"] for r in records}))
            project.update({"clips": records, "images": pictures})
            z.writestr("project.json", json.dumps(project, indent=1))
        os.replace(temp, out_path)
    except BaseException:
        try:
            os.remove(temp)
        except OSError:
            pass
        raise
    return project


def pack(edit, clips, clip_path, out_path, images=(), image_path=None):
    """Write the edit, the videos it uses and their reference pictures to out_path (a .zip).

    clips: id -> clip record. images: every image record; image_path(image) is its file.
    """
    used = []
    for piece in edit.get("pieces") or []:
        if piece["clip"] in clips and piece["clip"] not in used:
            used.append(piece["clip"])

    def head(kept):
        return {"kind": KIND,
                "edit": {"name": edit.get("name") or "Untitled edit", "ratio": edit.get("ratio") or "16:9",
                         "muted": bool(edit.get("muted")),
                         "pieces": [p for p in edit.get("pieces") or [] if p["clip"] in kept]}}

    project = _write(out_path, used, clips, clip_path, images, image_path, head)
    return {"clips": len(project["clips"]), "pieces": len(project["edit"]["pieces"]), "images": len(project["images"])}


def pack_clip(clip, clip_path, out_path, images=(), image_path=None, clips=None):
    """Write one video, the videos it was made from (from clips: id -> clip record) and its references to out_path."""
    if not os.path.exists(clip_path(clip)):
        raise Bad("This video's file isn't there any more.")
    project = _write(out_path, [clip["id"]], dict(clips or {}, **{clip["id"]: clip}), clip_path, images, image_path,
                     lambda kept: {"kind": CLIP_KIND})
    return {"clips": len(project["clips"]), "images": len(project["images"])}


def read(zip_path):
    """The project.json of a project zip, checked."""
    try:
        with zipfile.ZipFile(zip_path) as z:
            try:
                project = json.loads(z.read("project.json"))
            except KeyError:
                raise Bad("This zip isn't a MiniMax H3 Vision project: it has no project.json.")
            listed = set(z.namelist())
    except zipfile.BadZipFile:
        raise Bad("This file isn't a zip, or it is damaged.")
    except ValueError:
        raise Bad("The project.json in this zip can't be read.")
    if project.get("kind") not in (KIND, CLIP_KIND):
        raise Bad("This zip isn't a MiniMax H3 Vision video or edit project.")
    if int(project.get("version", 0)) > VERSION:
        raise Bad("This project was saved by a newer MiniMax H3 Vision. Update the plugin to open it.")
    for record in (project.get("clips") or []) + (project.get("images") or []):
        if record.get("zip_file") not in listed:
            raise Bad("The project lists %s, but it isn't in the zip." % record.get("zip_file"))
    return project


def unpack_images(z, project, images, images_dir, image_path, ids=None):
    """Bring a project's reference pictures in. Returns (new image records, {old name lower: name here}, how many were here).

    A picture whose id is already here with a file of the same size is the same picture. Others get
    a new id, and their name, or the name with a number after it when another picture has it.
    """
    added, names, same = [], {}, 0
    by_id = {i["id"]: i for i in images}
    taken = {i["name"] for i in images}
    os.makedirs(images_dir, exist_ok=True)
    for record in project.get("images") or []:
        info = z.getinfo(record["zip_file"])
        mine = by_id.get(record.get("id"))
        if mine and os.path.exists(image_path(mine)) and os.path.getsize(image_path(mine)) == info.file_size:
            names[record["name"].lower()] = mine["name"]
            if ids is not None:
                ids[record.get("id")] = mine
            same += 1
            continue
        image_id = record["id"] if record.get("id") and record["id"] not in by_id else "i" + uuid.uuid4().hex[:10]
        ext = os.path.splitext(record["zip_file"])[1].lower() or ".png"
        file = image_id + ext
        with z.open(info) as src, open(os.path.join(images_dir, file), "wb") as dst:
            shutil.copyfileobj(src, dst, 4 << 20)
        base = re.sub(r"[^A-Za-z0-9_]+", "_", record.get("name") or "image").strip("_")[:16] or "image"
        name, n = base, 2
        while name in taken:
            name, n = "%s%d" % (base, n), n + 1
        taken.add(name)
        image = {"id": image_id, "name": name, "file": file, "added": record.get("added") or 0,
                 **{k: record[k] for k in ("kind", "seconds", "width", "height", "audio") if k in record}}
        by_id[image_id] = image
        names[record["name"].lower()] = name
        if ids is not None:
            ids[record.get("id")] = image
        added.append(image)
    return added, names, same


def unpack(zip_path, existing, clips_dir, clip_path, images=(), images_dir=None, image_path=None):
    """Bring a project's videos and reference pictures in.

    Returns (edit, new clip records, how many videos were already here, new image records, how many
    pictures were already here). edit is None for a single video's zip.

    existing: id -> clip record already in the app. A video whose id is already here with a file of
    the same size is the same video, so it isn't copied again. Others get a new id if theirs is taken.
    A picture whose name had to change here has its @name changed in the new videos' prompts too.
    """
    project = read(zip_path)
    os.makedirs(clips_dir, exist_ok=True)
    added, same, ids = [], 0, {}
    with zipfile.ZipFile(zip_path) as z:
        new_images, names, same_images, image_ids = ([], {}, 0, {})
        if images_dir and image_path:
            new_images, names, same_images = unpack_images(z, project, list(images), images_dir, image_path, image_ids)
        for record in project.get("clips") or []:
            info = z.getinfo(record["zip_file"])
            mine = existing.get(record.get("id"))
            if mine and mine.get("file") and os.path.exists(clip_path(mine)) and os.path.getsize(clip_path(mine)) == info.file_size:
                ids[record["id"]] = mine["id"]
                same += 1
                continue
            name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(record["zip_file"])) or "clip.mp4"
            stem, ext = os.path.splitext(name)
            n = 2
            while os.path.exists(os.path.join(clips_dir, name)):
                name = "%s_%d%s" % (stem, n, ext)
                n += 1
            with z.open(info) as src, open(os.path.join(clips_dir, name), "wb") as dst:
                shutil.copyfileobj(src, dst, 4 << 20)
            clip_id = record.get("id") if record.get("id") and record.get("id") not in existing else "c" + uuid.uuid4().hex[:10]
            clip = {k: record[k] for k in CLIP_FIELDS if k in record}
            clip.update({"id": clip_id, "kind": "video", "status": "done", "step": record.get("steps", 0), "file": name})
            clip["prompt"] = rename_mentions(clip.get("prompt"), names)
            pictures = {i["name"].lower(): i for i in list(images) + new_images}
            if isinstance(clip.get("pictures"), list):
                # The Video tab's pictures, in order: by id (an id can change coming in), else by name.
                here = {i["id"]: i for i in list(images) + new_images}
                kept = []
                for p in clip["pictures"]:
                    image = image_ids.get(p.get("id")) or here.get(p.get("id")) or pictures.get(names.get(str(p.get("name")).lower(), "").lower())
                    if image:
                        kept.append({"id": image["id"], "name": image["name"], **({"kind": "video"} if image.get("kind") == "video" else {})})
                clip["pictures"] = clip["refs"] = kept
            else:
                clip["refs"] = [{"id": pictures[n.lower()]["id"], "name": pictures[n.lower()]["name"]}
                                for n in dict.fromkeys(MENTION.findall(clip["prompt"])) if n.lower() in pictures]
            clip.setdefault("starred", False)
            ids[record["id"]] = clip_id
            existing = dict(existing, **{clip_id: clip})
            added.append(clip)
    # The videos they were made from may have a new id here.
    for clip in added:
        for k in SOURCE_FIELDS:
            if isinstance(clip.get(k), dict) and clip[k].get("from") in ids:
                clip[k] = dict(clip[k], **{"from": ids[clip[k]["from"]]})
    if project.get("kind") == CLIP_KIND:
        return None, added, same, new_images, same_images
    edit = project.get("edit") or {}
    edit = {"name": str(edit.get("name") or "Untitled edit")[:80], "ratio": edit.get("ratio") if edit.get("ratio") in ("16:9", "9:16", "1:1") else "16:9",
            "muted": bool(edit.get("muted")),
            "pieces": [{"id": str(p["id"])[:40], "clip": ids[p["clip"]], "in": float(p["in"]), "out": float(p["out"])}
                       for p in edit.get("pieces") or [] if p.get("clip") in ids]}
    return edit, added, same, new_images, same_images
