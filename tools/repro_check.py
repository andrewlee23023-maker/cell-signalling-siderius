"""Local reproduction + inference check before spending an upload.

    python tools/repro_check.py SUBMISSION_DIR --data-dir DATA_DIR \
        [--package-dir TASK_PACKAGE] [--eval-split val] [--timeout 3600] [--report out.json]

1. Runs tools/package.py checks.
2. Copies the submission into a fresh work dir, DELETES the outputs that
   train_command must produce (weights/initial for model; plugin for algorithm),
   and builds a read-only train-only DATA_DIR (files whose name says val/test
   are excluded). Runs the exact train_command under bash with DATA_DIR,
   PACKAGE_DIR, OUTPUT_DIR, SEED set, with no network (bubblewrap) when available,
   under the retraining time limit.
3. Confirms the declared outputs were written and compares them with the staged ones.
4. Loads the plugin in a separate isolated Python process from a copy WITHOUT
   train/, runs inference on the eval split (or random inputs), checks the
   output contract, and times it against the 1,200 s scoring limit.
5. Reports provisional AP (sklearn) on the eval split if labels exist.

It never modifies SUBMISSION_DIR. It cannot prove the official scorer agrees;
it catches broken paths, missing outputs, wrong shapes and train/ imports.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_io import inventory, load_split  # noqa: E402
from package import SKIP, report as package_report  # noqa: E402

SCORING_LIMIT_S = 1200

CHILD = r'''
import json, math, sys, time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
root = Path(sys.argv[1]); X = np.load(sys.argv[2]); out_path = sys.argv[3]
sys.path.insert(0, str(root))
m = json.loads((root / "submission.json").read_text())
spec = spec_from_file_location("plugin_under_test", root / m["plugin"])
mod = module_from_spec(spec); spec.loader.exec_module(mod)
res = {"n": int(len(X))}
t0 = time.time()
if m["kind"] == "model":
    import torch
    cls = getattr(mod, m["symbol"]) if m.get("symbol") else getattr(mod, "PLUGIN_MODEL_CLASS", None)
    if cls is None:
        cands = [v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, torch.nn.Module) and v.__module__ == mod.__name__]
        if len(cands) != 1: raise SystemExit(f"cannot resolve the model class (symbol missing, {len(cands)} candidates)")
        cls = cands[0]
    cfg = json.loads((root / m["plugin_config"]).read_text()) if m.get("plugin_config") else {}
    model = cls(**cfg)
    model.load_state_dict(torch.load(root / m["weights"], map_location="cpu", weights_only=True), strict=True)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            o = model(torch.from_numpy(X[i:i+256]).float())
            if tuple(o.shape) != (len(X[i:i+256]), 1): raise SystemExit(f"model output shape {tuple(o.shape)}, expected [B, 1]")
            outs.append(o.numpy().reshape(-1))
    scores = np.concatenate(outs)
else:
    has_b, has_s = hasattr(mod, "predict_batch"), hasattr(mod, "predict")
    if has_b:
        scores = np.asarray(mod.predict_batch(X), dtype=np.float64).reshape(-1)
        if scores.shape != (len(X),): raise SystemExit(f"predict_batch returned shape {scores.shape}, expected [N]")
    else:
        scores = np.array([float(mod.predict(x)) for x in X])
    if has_s:
        one = mod.predict(X[0])
        if not isinstance(one, (float, int, np.floating, np.integer)): raise SystemExit(f"predict(x) returned {type(one).__name__}, expected a float")
        res["predict_matches_batch"] = bool(np.isclose(float(one), scores[0], equal_nan=True))
res["seconds"] = time.time() - t0
res["finite_fraction"] = float(np.isfinite(scores).mean())
res["unique_scores"] = int(len(np.unique(scores)))
np.save(out_path, scores)
print(json.dumps(res))
'''


def copy_tree(src: Path, dst: Path, exclude_top: set[str] = frozenset()) -> None:
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if rel.parts[0] in exclude_top or any(part in SKIP for part in rel.parts):
            continue
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif p.is_file() and not p.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)


def make_read_only(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True) + [root]:
        mode = p.stat().st_mode
        p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def make_writable(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        try:
            p.chmod(p.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("submission", type=Path)
    ap_.add_argument("--data-dir", type=Path, required=True)
    ap_.add_argument("--package-dir", type=Path)
    ap_.add_argument("--eval-split", default="val")
    ap_.add_argument("--timeout", type=float, default=3600)
    ap_.add_argument("--allow-network", action="store_true")
    ap_.add_argument("--keep", action="store_true", help="keep the temporary work dir for inspection")
    ap_.add_argument("--report", type=Path)
    ap_.add_argument("--cs-eval", action="store_true",
                     help="cell-signalling: evaluate on validation tensors from tools/cs_data.py with the official metric")
    a = ap_.parse_args()
    sub = a.submission.resolve()
    result: dict = {"submission": str(sub), "ok": False}

    print("== 1. package check")
    if not package_report(sub):
        return 1
    m = json.loads((sub / "submission.json").read_text())
    kind = m["kind"]
    outputs = [m["weights"], m["initial_weights"]] if kind == "model" else [m["plugin"]]

    tmp = Path(tempfile.mkdtemp(prefix="repro-cell-signalling-"))
    work, data, pkg, infer, home = (tmp / d for d in ("work", "train_data", "package", "infer", "home"))
    for d in (work, data, pkg, infer, home):
        d.mkdir()
    try:
        print(f"== 2. fresh work dir {work}")
        copy_tree(sub, work)
        for rel in outputs:
            (work / rel).unlink(missing_ok=True)
        print(f"   removed staged outputs so train_command must regenerate them: {outputs}")
        kept = []
        for f in {x.path for x in inventory(a.data_dir)} | {p for p in a.data_dir.rglob('*') if p.is_file()}:
            name = f.relative_to(a.data_dir).as_posix().lower()
            if any(t in name for t in ("val", "valid", "test", "dev")) and "train" not in name:
                continue
            (data / f.relative_to(a.data_dir)).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, data / f.relative_to(a.data_dir))
            kept.append(f.relative_to(a.data_dir).as_posix())
        print(f"   train-only DATA_DIR files: {sorted(kept)}")
        if a.package_dir:
            copy_tree(a.package_dir.resolve(), pkg)
        make_read_only(data)
        make_read_only(pkg)

        env = {
            "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
            "HOME": str(home), "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "DATA_DIR": str(data), "PACKAGE_DIR": str(pkg), "OUTPUT_DIR": str(work), "SEED": str(m["seed"]),
        }
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        cmd = ["bash", "-c", m["train_command"]]
        bwrap = shutil.which("bwrap")
        if bwrap and not a.allow_network:
            cmd = [bwrap, "--dev-bind", "/", "/", "--unshare-net", "--die-with-parent", "--chdir", str(work), "--", *cmd]
            print("   network: disabled (bubblewrap --unshare-net)")
        else:
            print("   network: NOT disabled" + ("" if bwrap else " (bwrap not found)"))
        print(f"== 3. train_command (limit {a.timeout:.0f}s): {m['train_command']}")
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=work, env=env, timeout=a.timeout, text=True, capture_output=True)
        except subprocess.TimeoutExpired:
            print(f"   FAIL: exceeded {a.timeout:.0f}s")
            return 1
        result["train_seconds"] = round(time.time() - t0, 1)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
        print("   | " + "\n   | ".join(tail))
        if proc.returncode:
            print(f"   FAIL: exit {proc.returncode}")
            return 1
        print(f"   exit 0 in {result['train_seconds']}s")
        missing = [rel for rel in outputs if not (work / rel).is_file()]
        if missing:
            print(f"   FAIL: train_command did not write {missing}")
            return 1
        same = {rel: (work / rel).read_bytes() == (sub / rel).read_bytes() for rel in outputs if (sub / rel).is_file()}
        if kind == "model":
            import torch

            for rel in outputs:
                sd_new = torch.load(work / rel, map_location="cpu", weights_only=True)
                if (sub / rel).is_file() and not same[rel]:
                    sd_old = torch.load(sub / rel, map_location="cpu", weights_only=True)
                    keys_ok = sd_new.keys() == sd_old.keys()
                    close = keys_ok and all(torch.allclose(sd_new[k].float(), sd_old[k].float(), atol=1e-5) for k in sd_new)
                    same[rel] = "allclose" if close else ("same keys, values differ" if keys_ok else "DIFFERENT KEYS")
        result["regenerated_vs_staged"] = same
        print(f"   regenerated vs staged: {same}")

        print("== 4. isolated inference (copy without train/)")
        copy_tree(work, infer, exclude_top={m["train_dir"].strip("/").split("/")[0]})
        cs = None
        try:
            if a.cs_eval:
                from cs_data import load as cs_load

                cs = cs_load("validation")
                X, y = cs["X"], cs["y"].astype(float)
            else:
                ev = load_split(a.data_dir, a.eval_split)
                X, y = ev.X, ev.y
            print(f"   eval split {a.eval_split!r}: X {X.shape}" + (f", positives {int(y.sum())}" if y is not None else ""))
        except Exception as exc:
            print(f"   no eval split ({exc.__class__.__name__}); using 64 random non-negative inputs")
            X, y = np.abs(np.random.default_rng(0).normal(size=(64, 5, 1024))).astype(np.float32), None
        xin, sout = tmp / "X.npy", tmp / "scores.npy"
        np.save(xin, X)
        child = subprocess.run([sys.executable, "-I", "-c", CHILD, str(infer), str(xin), str(sout)],
                               cwd=infer, env={**env, "HOME": str(home)}, text=True, capture_output=True, timeout=SCORING_LIMIT_S)
        if child.returncode:
            print("   FAIL:\n   | " + "\n   | ".join((child.stdout + child.stderr).strip().splitlines()[-15:]))
            return 1
        inf = json.loads(child.stdout.strip().splitlines()[-1])
        result["inference"] = inf
        per = inf["seconds"] / max(inf["n"], 1)
        print(f"   ok: {inf['n']} samples in {inf['seconds']:.2f}s ({per * 1000:.2f} ms/sample), finite {inf['finite_fraction']:.3f}, "
              f"unique scores {inf['unique_scores']}" + (f", predict==batch[0]: {inf['predict_matches_batch']}" if "predict_matches_batch" in inf else ""))
        if inf["finite_fraction"] < 1:
            print("   WARN: non-finite scores")
        if inf["unique_scores"] < max(2, inf["n"] // 20):
            print("   WARN: many tied scores; check how lr_auprc breaks ties")
        print(f"   {SCORING_LIMIT_S}s scoring limit allows ~{int(SCORING_LIMIT_S / max(per, 1e-9)):,} samples at this speed (CPU/GPU here may differ)")
        if cs is not None:
            from cs_data import score as cs_score

            agg = cs_score(cs, np.load(sout))["aggregate"]
            result["official_validation"] = agg
            print(f"== 5. OFFICIAL metric on validation: lr_auprc {agg['lr_auprc']:.4f}, "
                  f"direction_accuracy {agg['direction_accuracy']:.4f}, lr_auroc {agg['lr_auroc']:.4f}")
        elif y is not None:
            from sklearn.metrics import average_precision_score

            scores = np.load(sout)
            result["provisional_ap"] = float(average_precision_score(y, np.nan_to_num(scores, nan=-1e9)))
            print(f"== 5. provisional AP on {a.eval_split}: {result['provisional_ap']:.4f} (prior {y.mean():.4f}) — not the official lr_auprc")
        result["ok"] = True
        print("\nREPRO CHECK PASSED (local). Still: official scoring may differ; keep the zip sha256.")
        return 0
    finally:
        if a.report:
            a.report.write_text(json.dumps(result, indent=2))
        if a.keep:
            print(f"kept {tmp}")
        else:
            make_writable(tmp)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
