"""The Models page: which of model_catalog.json's groups are installed, and downloading them in the background.

Downloads run one file at a time in a child Python process (huggingface_hub's hf_hub_download, with hf_xet when it
is installed, like the RunPod template's start.sh), into models/.downloads, then move into place. Progress is the
size of what has arrived so far. Gated repositories (Klein) need a Hugging Face token: HF_TOKEN from the pod, or one
pasted on the Models page, kept until ComfyUI restarts or, if asked, in models/.downloads/hf_token on the volume.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid

import folder_paths

LOG = logging.getLogger("minimax_flow")
CATALOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_catalog.json")

# One file in the child process: find it by name anywhere in the repository, download it, print where it went.
FETCH = r"""
import sys
from huggingface_hub import HfApi, hf_hub_download
repo, name, stage = sys.argv[1:4]
paths = [p for p in HfApi().list_repo_files(repo) if p == name or p.endswith("/" + name)]
if not paths:
    sys.exit("NOTFOUND %s isn't in %s" % (name, repo))
print(hf_hub_download(repo, sorted(paths, key=len)[0], local_dir=stage), flush=True)
"""
# Who a token belongs to (it checks the token), in the child process too.
WHOAMI = r"""
import json, os
from huggingface_hub import HfApi
info = HfApi(token=os.environ["MMFLOW_CHECK_TOKEN"]).whoami()
print(json.dumps({"name": info.get("name") or info.get("fullname") or "?"}))
"""
# The sizes of a repository's files, in the child process too (it needs the network).
SIZES = r"""
import json, sys
from huggingface_hub import HfApi
repo, names = sys.argv[1], sys.argv[2:]
api = HfApi()
paths = {}
for p in api.list_repo_files(repo):
    for n in names:
        if p == n or p.endswith("/" + n):
            paths.setdefault(n, p)
info = {i.path: i.size for i in api.get_paths_info(repo, list(paths.values()))}
print(json.dumps({n: info.get(p) for n, p in paths.items()}))
"""


# ---- The Hugging Face token ---------------------------------------------------
_PASTED = {"token": None, "user": None}


def token_file():
    return os.path.join(models_dir(), ".downloads", "hf_token")


def token():
    """The token downloads use: the pod's HF_TOKEN, else the one pasted on the Models page, else the remembered one."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if _PASTED["token"]:
        return _PASTED["token"]
    try:
        with open(token_file(), encoding="utf-8") as f:
            saved = f.read().strip()
        return saved or None
    except OSError:
        return None


def token_state():
    """Where the token comes from ("pod", "pasted", "saved" or None), and whose it is when known."""
    if os.environ.get("HF_TOKEN"):
        return {"from": "pod", "user": None}
    if os.path.isfile(token_file()):
        return {"from": "saved", "user": _PASTED["user"]}
    if _PASTED["token"]:
        return {"from": "pasted", "user": _PASTED["user"]}
    return {"from": None, "user": None}


def child_env():
    env = {**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    t = token()
    if t:
        env["HF_TOKEN"] = t
    return env


async def set_token(value, remember):
    """Check a pasted token with Hugging Face and use it; with remember, also keep it on the volume. ValueError if
    Hugging Face doesn't take it."""
    value = (value or "").strip()
    if not value.startswith("hf_") or len(value) < 20 or any(c.isspace() for c in value):
        raise ValueError("That isn't a Hugging Face token: they start with hf_.")
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", WHOAMI, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                env={**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", "MMFLOW_CHECK_TOKEN": value})
    out, err = await proc.communicate()
    if proc.returncode != 0:
        text = err.decode("utf-8", "replace")
        if "401" in text or "Invalid" in text or "Unauthorized" in text:
            raise ValueError("Hugging Face turned that token down: check it, or make a new read token.")
        raise ValueError("Couldn't reach Hugging Face to check the token: " + (text.strip().splitlines() or ["unknown error"])[-1][:200])
    _PASTED.update(token=value, user=json.loads(out.decode().strip().splitlines()[-1]).get("name"))
    path = token_file()
    if remember:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value)
    elif os.path.isfile(path):
        os.remove(path)
    return token_state()


def forget_token():
    _PASTED.update(token=None, user=None)
    try:
        os.remove(token_file())
    except OSError:
        pass
    return token_state()


def catalog():
    with open(CATALOG_FILE, encoding="utf-8") as f:
        return json.load(f)["groups"] + [custom_group(m) for m in custom()]


# ---- Video models added from a Hugging Face link (kept on the volume, next to the models) ----
SAMPLERS = ["euler", "lcm", "res_multistep"]
SCHEDULERS = ["simple", "beta", "beta57"]


def custom_file():
    return os.path.join(models_dir(), "mmvision_custom_models.json")


def custom():
    try:
        with open(custom_file(), encoding="utf-8") as f:
            return [m for m in json.load(f) if isinstance(m, dict) and m.get("id") and m.get("file")]
    except (OSError, ValueError):
        return []


def save_custom(rows):
    path = custom_file()
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    os.replace(temp, path)
    register_custom()


def custom_group(m):
    return {"id": "custom-" + m["id"], "tab": "video", "name": m["name"], "needs": ["h3-base"], "custom": True, "url": m.get("url"),
            "about": "Added from a link: %s steps, %s / %s." % (m["steps"], m["sampler"], m["scheduler"]),
            "files": [[m["repo"], m["file"], "diffusion_models"]]}


def register_custom():
    """Put the added models among the video models (h3_graph's families), as custom_<id>."""
    from . import h3_graph
    for key in [k for k in h3_graph.PROFILES if k.startswith(h3_graph.CUSTOM)]:
        del h3_graph.PROFILES[key]
    for m in custom():
        h3_graph.PROFILES[h3_graph.CUSTOM + m["id"]] = {"title": m["name"], "unet": [m["file"]], "steps": int(m["steps"]),
                                                        "sampler": m["sampler"], "scheduler": m["scheduler"], "custom": True}


def parse_link(url):
    """(repo, path in the repo) from a Hugging Face file link: .../owner/repo/blob|resolve/revision/path."""
    import re
    m = re.match(r"^\s*https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([\w.-]+/[\w.-]+)/(?:blob|resolve)/[^/]+/(.+?)(?:\?.*)?\s*$", url or "")
    if not m:
        raise ValueError("That isn't a link to a file on Hugging Face: open the file there and copy its address "
                         "(huggingface.co/owner/repo/blob/main/name.safetensors).")
    path = m.group(2)
    if not path.endswith(".safetensors"):
        raise ValueError("The link must be to a .safetensors file.")
    return m.group(1), path


async def add_custom(url, name, steps, sampler, scheduler):
    """Check the link with Hugging Face, add the model to the list and return its group id (to download)."""
    from . import h3_graph
    repo, path = parse_link(url)
    file = path.split("/")[-1]
    taken = h3_graph.profile_for(file)
    if taken:
        raise ValueError("%s is already in the list, as %s." % (file, h3_graph.PROFILES[taken]["title"]))
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = 0
    if not 1 <= steps <= 60:
        raise ValueError("Steps go from 1 to 60.")
    if sampler not in SAMPLERS or scheduler not in SCHEDULERS:
        raise ValueError("Pick a sampler and a scheduler from the list.")
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", SIZES, repo, file, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE, env=child_env())
    out, err = await proc.communicate()
    if proc.returncode != 0:
        text = err.decode("utf-8", "replace")
        if any(k in text for k in ("ProxyError", "ConnectError", "ConnectionError", "Temporary failure", "timed out")):
            raise ValueError("Couldn't reach Hugging Face: " + (text.strip().splitlines() or ["unknown error"])[-1][:200])
        if any(k in text for k in ("401", "403", "Gated", "gated")):
            raise ValueError("Hugging Face won't show %s: it's gated or private. Paste a token at the top of this page, and accept its license on Hugging Face." % repo)
        if "404" in text or "RepositoryNotFound" in text:
            raise ValueError("There's no repository %s on Hugging Face." % repo)
        raise ValueError("Couldn't reach Hugging Face: " + (text.strip().splitlines() or ["unknown error"])[-1][:200])
    size = json.loads(out.decode().strip().splitlines()[-1]).get(file)
    if not size:
        raise ValueError("%s isn't in %s." % (file, repo))
    rows = custom()
    base = "".join(c if c.isalnum() else "_" for c in file[:-12].lower())[:48].strip("_") or "model"
    key, n = base, 2
    while any(r["id"] == key for r in rows):
        key, n = "%s_%d" % (base, n), n + 1
    name = (name or "").strip()[:80] or file[:-12]
    rows.append({"id": key, "name": name, "url": url.strip(), "repo": repo, "file": file, "steps": steps, "sampler": sampler, "scheduler": scheduler})
    save_custom(rows)
    STORE.sizes[repo + "/" + file] = size
    return "custom-" + key


def remove_custom(group_id):
    """Take an added model off the list (its file is deleted separately, with Delete)."""
    key = group_id[len("custom-"):]
    rows = custom()
    if not any(r["id"] == key for r in rows):
        raise KeyError(group_id)
    save_custom([r for r in rows if r["id"] != key])


def models_dir():
    return folder_paths.models_dir


def find(name, folder):
    """Where ComfyUI has this file (any folder for that kind of model, any subfolder), or None."""
    kind = folder.split("/")[0]
    direct = os.path.join(models_dir(), folder, name)
    if os.path.isfile(direct) and os.path.getsize(direct) > 0:
        return direct
    try:
        for f in folder_paths.get_filename_list(kind):
            if os.path.basename(f.replace("\\", "/")) == name:
                path = folder_paths.get_full_path(kind, f)
                if path and os.path.getsize(path) > 0:
                    return path
    except Exception:
        pass
    return None


def refresh(folder):
    """ComfyUI caches its model lists and only notices a change in the top folder: forget that one."""
    getattr(folder_paths, "filename_list_cache", {}).pop(folder.split("/")[0], None)


class Store:
    def __init__(self):
        self.queue = []          # group ids waiting
        self.job = None          # the download running: {group, file, index, count, done, total, speed, started}
        self.errors = {}         # group id: why its last download stopped
        self.proc = None
        self.task = None
        self.sizes = {}          # "repo/name": bytes
        self.sizes_file = None
        self._sizes_started = False
        self.on_done = None      # called (group id) when a group is installed

    # ---- Sizes (looked up once, kept in the output folder) -----------------
    def load_sizes(self, out_dir):
        self.sizes_file = os.path.join(out_dir, "model_sizes.json")
        try:
            with open(self.sizes_file, encoding="utf-8") as f:
                self.sizes.update(json.load(f))
        except (OSError, ValueError):
            pass

    def start_sizes(self):
        if self._sizes_started:
            return
        self._sizes_started = True
        threading.Thread(target=self._fill_sizes, daemon=True).start()

    def _fill_sizes(self):
        by_repo = {}
        for g in catalog():
            for repo, name, _ in g["files"]:
                if repo + "/" + name not in self.sizes:
                    by_repo.setdefault(repo, set()).add(name)
        import subprocess
        for repo, names in by_repo.items():
            try:
                out = subprocess.run([sys.executable, "-c", SIZES, repo, *sorted(names)], capture_output=True, text=True, timeout=60, env=child_env())
                for name, size in json.loads(out.stdout.strip().splitlines()[-1]).items():
                    if size:
                        self.sizes[repo + "/" + name] = size
            except Exception as error:
                LOG.info("MiniMax H3 Vision couldn't look up the sizes in %s: %s", repo, error)
        if self.sizes_file:
            try:
                with open(self.sizes_file, "w", encoding="utf-8") as f:
                    json.dump(self.sizes, f)
            except OSError:
                pass

    # ---- What the page shows ----------------------------------------------
    def status(self):
        self.start_sizes()
        groups = []
        for g in catalog():
            files = []
            for repo, name, folder in g["files"]:
                path = find(name, folder)
                size = os.path.getsize(path) if path else self.sizes.get(repo + "/" + name)
                files.append({"repo": repo, "name": name, "folder": folder, "installed": bool(path), "size": size})
            have = sum(f["installed"] for f in files)
            state = "installed" if have == len(files) else "partly" if have else "missing"
            if self.job and self.job["group"] == g["id"]:
                state = "downloading"
            elif g["id"] in self.queue:
                state = "queued"
            groups.append({**{k: g.get(k) for k in ("id", "tab", "name", "about", "env", "gated", "needs", "custom", "url")},
                           "files": files, "state": state, "error": self.errors.get(g["id"]),
                           "missingBytes": sum(f["size"] or 0 for f in files if not f["installed"]),
                           "unknownSize": any(not f["installed"] and not f["size"] for f in files),
                           "bytes": sum(f["size"] or 0 for f in files)})
        usage = shutil.disk_usage(models_dir())
        return {"groups": groups, "job": self.job, "queue": list(self.queue),
                "disk": {"free": usage.free, "total": usage.total}, "token": token_state(),
                "add": {"samplers": SAMPLERS, "schedulers": SCHEDULERS}}

    # ---- Downloads -----------------------------------------------------------
    def download(self, group_id):
        if not any(g["id"] == group_id for g in catalog()):
            raise KeyError(group_id)
        self.errors.pop(group_id, None)
        if group_id not in self.queue and not (self.job and self.job["group"] == group_id):
            self.queue.append(group_id)
        if not self.task or self.task.done():
            self.task = asyncio.ensure_future(self._run())

    def cancel(self, group_id):
        if group_id in self.queue:
            self.queue.remove(group_id)
        if self.job and self.job["group"] == group_id:
            self.job["cancelled"] = True
            if self.proc and self.proc.returncode is None:
                self.proc.kill()

    def delete(self, group_id):
        """Delete the group's files that no other installed group uses. Returns the names kept because they are shared."""
        groups = catalog()
        group = next(g for g in groups if g["id"] == group_id)
        if self.job and self.job["group"] == group_id:
            raise RuntimeError("It's downloading: cancel it first.")
        others = set()
        for g in groups:
            if g["id"] != group_id and all(find(n, f) for _, n, f in g["files"]):
                others.update(n for _, n, _ in g["files"])
        kept, removed = [], []
        for _, name, folder in group["files"]:
            if name in others:
                kept.append(name)
                continue
            path = find(name, folder)
            if path:
                os.remove(path)
                removed.append(name)
                refresh(folder)
        return {"removed": removed, "kept": kept}

    async def _run(self):
        while self.queue:
            group_id = self.queue.pop(0)
            group = next((g for g in catalog() if g["id"] == group_id), None)
            if not group:
                continue
            todo = [(r, n, f) for r, n, f in group["files"] if not find(n, f)]
            self.job = {"group": group_id, "count": len(todo), "index": 0, "file": None, "done": 0, "total": None,
                        "speed": 0, "started": time.time()}
            try:
                for i, (repo, name, folder) in enumerate(todo):
                    self.job.update({"index": i + 1, "file": name, "done": 0, "total": self.sizes.get(repo + "/" + name), "speed": 0})
                    await self._fetch(repo, name, folder)
                    if self.job.get("cancelled"):
                        break
                if self.job.get("cancelled"):
                    self.errors[group_id] = "Cancelled."
                elif self.on_done:
                    self.on_done(group_id)
            except Exception as error:
                self.errors[group_id] = str(error)
                LOG.warning("MiniMax H3 Vision couldn't download %s: %s", group["name"], error)
            finally:
                self.job = None
                self.proc = None

    async def _fetch(self, repo, name, folder):
        stage = os.path.join(models_dir(), ".downloads", uuid.uuid4().hex[:8])
        os.makedirs(stage, exist_ok=True)
        try:
            self.proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", FETCH, repo, name, stage,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=child_env())
            watch = asyncio.ensure_future(self._watch(stage))
            out, err = await self.proc.communicate()
            watch.cancel()
            if self.job.get("cancelled"):
                return
            if self.proc.returncode != 0:
                raise RuntimeError(self._why(repo, name, err.decode("utf-8", "replace")))
            got = out.decode().strip().splitlines()[-1]
            dest = os.path.join(models_dir(), folder, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(got, dest)
            refresh(folder)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    async def _watch(self, stage):
        last, at = 0, time.time()
        while True:
            await asyncio.sleep(0.5)
            done = 0
            for root, _, files in os.walk(stage):
                for f in files:
                    try:
                        done += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            now = time.time()
            if self.job:
                speed = (done - last) / max(now - at, 1e-3)
                self.job["speed"] = 0.7 * self.job["speed"] + 0.3 * speed if self.job["speed"] else speed
                self.job["done"] = done
            last, at = done, now

    @staticmethod
    def _why(repo, name, err):
        tail = err.strip().splitlines()[-1] if err.strip() else "unknown error"
        if "NOTFOUND" in err:
            return "%s isn't in %s any more." % (name, repo)
        if any(k in err for k in ("GatedRepo", "401", "403", "Access to model", "gated")):
            if not token():
                return ("%s is gated: paste a Hugging Face read token at the top of this page (or set HF_TOKEN on the pod), "
                        "and accept the license at https://huggingface.co/%s." % (repo, repo))
            return "Hugging Face refused %s: accept the license at https://huggingface.co/%s with the account of your token." % (repo, repo)
        if "No space left" in err:
            return "The disk is full."
        return "Download failed: %s" % tail[:300]


STORE = Store()
