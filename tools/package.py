"""Check and zip a cell-signalling submission folder.

    python tools/package.py check SUBMISSION_DIR
    python tools/package.py zip   SUBMISSION_DIR --out uploads/NAME.zip

`check` validates submission.json and the staged files against the Track 2
contract (docs/track2-guide.md). `zip` re-runs `check`, then writes a NEW
archive of the folder's contents (submission.json at the ZIP root), lists it
and prints its size and sha256. It refuses to overwrite an existing archive.

Exit status: 0 ok, 1 errors found. Warnings never fail the check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

MAX_ZIP, MAX_UNPACKED, MAX_FILES = 100 * 2**20, 2**30, 2000
SEED_MAX = 4_294_967_295
SKIP = {"__pycache__", ".DS_Store", ".ipynb_checkpoints", ".pytest_cache"}
SECRET_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "Anthropic API key"),
    (re.compile(r"sk-(proj-)?[A-Za-z0-9]{20,}"), "OpenAI-style API key"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "Google API key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"(?i)(api[_-]?key|access[_-]?code|secret|password)\s*[=:]\s*['\"][^'\"]{8,}"), "credential assignment"),
]
DATA_SUFFIXES = {".npy", ".npz", ".h5", ".hdf5", ".parquet", ".feather", ".csv", ".tsv", ".pkl"}
MODEL_KEYS = {"kind", "seed", "initial_weights", "plugin", "symbol", "weights", "plugin_config", "train_command", "train_dir"}
ALGO_KEYS = {"kind", "seed", "initial_artifact", "plugin", "plugin_config", "train_command", "train_dir"}


def staged_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if not any(part in SKIP for part in p.relative_to(root).parts) and (p.is_file() or p.is_symlink()))


def check(root: Path) -> tuple[list[str], list[str], dict]:
    errors, warnings = [], []
    mpath = root / "submission.json"
    if not mpath.is_file():
        return [f"{mpath} is missing (it must sit at the ZIP root)"], warnings, {}
    try:
        m = json.loads(mpath.read_text())
    except json.JSONDecodeError as exc:
        return [f"submission.json is not valid JSON: {exc}"], warnings, {}
    if not isinstance(m, dict):
        return ["submission.json must be a JSON object"], warnings, {}

    kind = m.get("kind")
    seed = m.get("seed")
    if kind not in {"model", "algorithm"}:
        errors.append(f"kind must be 'model' or 'algorithm', got {kind!r}")
    if not (isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= SEED_MAX):
        errors.append(f"seed must be an integer 0..{SEED_MAX}, got {seed!r}")

    def need_file(key: str):
        v = m.get(key)
        if not isinstance(v, str) or not v:
            errors.append(f"'{key}' is required")
            return None
        if v.startswith("/") or ".." in Path(v).parts:
            errors.append(f"'{key}' must be a relative path inside the package, got {v!r}")
            return None
        if not (root / v).is_file():
            errors.append(f"'{key}' names {v!r}, which does not exist")
            return None
        return root / v

    if kind == "model":
        unknown = set(m) - MODEL_KEYS
        need_file("plugin"), need_file("weights"), need_file("initial_weights")
        for key in ("weights", "initial_weights"):
            p = root / str(m.get(key, ""))
            if p.is_file():
                try:
                    import torch

                    sd = torch.load(p, map_location="cpu", weights_only=True)
                    if not (isinstance(sd, dict) and sd and all(isinstance(v, torch.Tensor) for v in sd.values())):
                        errors.append(f"{key} ({p.name}) is not a non-empty state_dict of tensors")
                except Exception as exc:
                    errors.append(f"{key} ({p.name}) does not load with torch.load(weights_only=True): {exc}")
        w, iw = root / str(m.get("weights", "")), root / str(m.get("initial_weights", ""))
        if w.is_file() and iw.is_file() and w.read_bytes() == iw.read_bytes():
            warnings.append("weights and initial_weights are byte-identical: was the model trained?")
    elif kind == "algorithm":
        unknown = set(m) - ALGO_KEYS
        for key in ("weights", "initial_weights"):
            if key in m:
                errors.append(f"algorithm kind must omit '{key}'")
        need_file("plugin"), need_file("initial_artifact")
        stray = [p.relative_to(root).as_posix() for p in staged_files(root) if p.suffix in {".pt", ".pth", ".bin", ".safetensors"}]
        if stray:
            errors.append(f"algorithm kind must not ship weight files (no dummy weights): {stray}")
    else:
        unknown = set()
    if unknown:
        warnings.append(f"unrecognized manifest keys (check the guide): {sorted(unknown)}")
    if m.get("plugin_config"):
        cfg = need_file("plugin_config")
        if cfg:
            try:
                json.loads(cfg.read_text())
            except json.JSONDecodeError as exc:
                errors.append(f"plugin_config is not valid JSON: {exc}")

    tc = m.get("train_command")
    if not isinstance(tc, str) or not tc.strip():
        errors.append("'train_command' is required")
    else:
        if re.search(r"(^|\s)/(home|Users|mnt|tmp|root)/", tc):
            errors.append(f"train_command uses an absolute path from this machine: {tc!r}")
        if "$DATA_DIR" not in tc and "${DATA_DIR}" not in tc:
            warnings.append("train_command does not mention $DATA_DIR")
        if "$SEED" not in tc and "${SEED}" not in tc:
            warnings.append("train_command does not pass $SEED; the manifest seed must drive every RNG")
    td = m.get("train_dir")
    if not isinstance(td, str) or not (root / td).is_dir():
        errors.append(f"'train_dir' must name an existing folder, got {td!r}")
    elif not any(p.suffix == ".py" for p in (root / td).rglob("*.py")):
        errors.append(f"train_dir {td!r} contains no Python source")

    plugin = root / str(m.get("plugin", ""))
    if plugin.is_file():
        text = plugin.read_text()
        bad = re.findall(r"^\s*(?:from|import)\s+(train|data_io)\b", text, flags=re.M)
        if bad:
            errors.append(f"plugin imports training-side modules {bad}; inference cannot import train/ files")
        if kind == "algorithm" and not re.search(r"^def (predict|predict_batch)\s*\(", text, flags=re.M):
            errors.append("algorithm plugin defines neither predict(x) nor predict_batch(X)")

    files = staged_files(root)
    links = [p.relative_to(root).as_posix() for p in files if p.is_symlink()]
    if links:
        errors.append(f"symlinks are not allowed: {links}")
    regular = [p for p in files if not p.is_symlink()]
    total = sum(p.stat().st_size for p in regular)
    if len(regular) > MAX_FILES:
        errors.append(f"{len(regular)} files > {MAX_FILES}")
    if total > MAX_UNPACKED:
        errors.append(f"unpacked size {total} B > 1 GiB")
    for p in regular:
        rel = p.relative_to(root).as_posix()
        if p.suffix.lower() in DATA_SUFFIXES and p.stat().st_size > 1_000_000:
            warnings.append(f"{rel} looks like a data file ({p.stat().st_size} B); data is supplied via $DATA_DIR")
        if p.suffix.lower() in {".log", ".out"} or rel.startswith(("logs/", "runs/", "wandb/")):
            warnings.append(f"{rel} looks like a log/run artifact; inspect it for secrets or drop it")
        if p.stat().st_size < 5_000_000:
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            for pat, what in SECRET_PATTERNS:
                if pat.search(text):
                    errors.append(f"{rel} appears to contain a {what}; remove it")
    for env in (".env", "credentials.json"):
        if (root / env).exists():
            errors.append(f"{env} must never be staged")
    return errors, warnings, {"kind": kind, "seed": seed, "files": len(regular), "bytes": total}


def report(root: Path) -> bool:
    errors, warnings, info = check(root)
    print(f"check {root}: kind={info.get('kind')} seed={info.get('seed')} files={info.get('files')} bytes={info.get('bytes')}")
    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  ERROR {e}")
    print("  OK" if not errors else f"  {len(errors)} error(s)")
    return not errors


def make_zip(root: Path, out: Path) -> int:
    if not report(root):
        return 1
    if out.exists():
        print(f"refusing to overwrite {out}: zip would merge stale files. Choose a new name.")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in staged_files(root):
            z.write(p, p.relative_to(root).as_posix())
    size = out.stat().st_size
    names = zipfile.ZipFile(out).namelist()
    print(f"\nwrote {out}")
    for n in names:
        print(f"  {n}")
    if "submission.json" not in names:
        print("ERROR submission.json is not at the ZIP root")
        return 1
    if size > MAX_ZIP:
        print(f"ERROR zip is {size} B > 100 MiB")
        return 1
    print(f"size {size} bytes\nsha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("dir", type=Path)
    z = sub.add_parser("zip")
    z.add_argument("dir", type=Path)
    z.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "check":
        return 0 if report(a.dir.resolve()) else 1
    return make_zip(a.dir.resolve(), a.out.resolve())


if __name__ == "__main__":
    sys.exit(main())
