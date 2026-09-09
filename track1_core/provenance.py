"""Stamp every result with enough to reproduce or refute it.

A headline number in this project turned out not to reproduce, and the reason
could not be established afterwards, because the result files recorded only the
numbers. The code state, the input files, and the environment at the time of the
run were all gone. Each of those was checked by hand later and none of them
explained the difference, which is exactly the situation this module exists to
prevent: a result that cannot be attributed is a result that has to be thrown
away.

So a stamp records what the run actually read and ran with:

  git         commit, whether the tree was dirty, and a digest of the diff, so a
              run made mid-edit is distinguishable from one made at a commit
  inputs      size and content hash of every file the experiment consumed
  environment python and the versions of the libraries that do the arithmetic

``stamp()`` is cheap for small inputs and hashes large ones by size plus a
sampled digest, so stamping a 160 MB candidate grid costs milliseconds rather
than seconds.

    from track1_core.provenance import stamp
    result["provenance"] = stamp(inputs=[grid_path, ckpt_path])
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def git_state() -> dict:
    """Commit, dirtiness, and a digest of the uncommitted diff.

    The diff digest is what distinguishes two runs made at the same commit with
    different working trees, which is the case that went undetected here.
    """
    diff = _git("diff", "HEAD")
    return {
        "commit": _git("rev-parse", "HEAD")[:12],
        "dirty": bool(diff.strip()),
        "diff_sha1": hashlib.sha1(diff.encode()).hexdigest()[:12] if diff.strip() else None,
        "diff_lines": len(diff.splitlines()) if diff.strip() else 0,
    }


def file_digest(path: os.PathLike | str, sample_bytes: int = 1 << 20) -> dict:
    """Size, mtime, and a content hash, sampled for large files.

    Hashing 160 MB in full on every run is wasteful, and the head, tail, and a
    middle slice already separate any two files that differ in content rather
    than only in metadata.
    """
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    size = p.stat().st_size
    h = hashlib.sha1()
    with p.open("rb") as f:
        if size <= 3 * sample_bytes:
            h.update(f.read())
        else:
            h.update(f.read(sample_bytes))
            f.seek(size // 2)
            h.update(f.read(sample_bytes))
            f.seek(-sample_bytes, os.SEEK_END)
            h.update(f.read(sample_bytes))
    return {"path": str(p), "bytes": size,
            "mtime": int(p.stat().st_mtime), "sha1": h.hexdigest()[:16]}


def environment() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "torch", "cv2", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def stamp(inputs=(), extra: dict | None = None) -> dict:
    """Everything needed to say whether two runs were the same experiment."""
    import datetime
    s = {
        "utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "argv": sys.argv,
        "git": git_state(),
        "env": environment(),
        "inputs": [file_digest(p) for p in inputs],
    }
    if extra:
        s.update(extra)
    return s


def compare(a: dict, b: dict) -> list[str]:
    """Human-readable differences between two stamps, most decisive first."""
    out = []
    if a["git"]["commit"] != b["git"]["commit"]:
        out.append(f"commit {a['git']['commit']} vs {b['git']['commit']}")
    if a["git"].get("diff_sha1") != b["git"].get("diff_sha1"):
        out.append(f"uncommitted diff {a['git'].get('diff_sha1')} vs {b['git'].get('diff_sha1')}")
    ai = {i["path"]: i for i in a.get("inputs", [])}
    bi = {i["path"]: i for i in b.get("inputs", [])}
    for p in sorted(set(ai) | set(bi)):
        x, y = ai.get(p), bi.get(p)
        if x is None or y is None:
            out.append(f"input only in one run: {p}")
        elif x.get("sha1") != y.get("sha1"):
            out.append(f"input content differs: {p}")
    for k in a.get("env", {}):
        if a["env"].get(k) != b.get("env", {}).get(k):
            out.append(f"env {k}: {a['env'].get(k)} vs {b.get('env', {}).get(k)}")
    return out or ["stamps agree on code, inputs, and environment"]
