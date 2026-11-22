"""Prepare (offline) and submit one SIDERIUS workflow run for cell-signalling.

    python project/scripts/launch.py --workflow experiment/workflow-smoke.py --profile config/llm-smoke.json \
        --run-name smoke --estimate-usd 0.05

Run with the SIDERIUS-wsl interpreter (GPU training on this WSL laptop) and with
the provider keys in the environment (setup-inputs/with_keys.sh). Steps:
  1. spend guard: refuse if any budgeted provider would pass its stop threshold
     after adding --estimate-usd for this run's main provider;
  2. prepare the run offline (snapshot of workflow, task, profile);
  3. build the launch request and submit it; record everything in results/launches.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[1]
W = PROJECT.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from spend_guard import tally  # noqa: E402

from tools.candidate_operations import OperationStoreConfig, build_reusable_operation  # noqa: E402
from tools.candidate_prepare import ReusableCandidatePreparationRequest, select_reusable_preparation  # noqa: E402

import os  # noqa: E402

RESULTS = PROJECT / "results"
RUNS = RESULTS / "runs"
# One active operation per store: STORE_NAME=store2 gives a parallel queue its own operations store.
STORE_NAME = os.environ.get("STORE_NAME", "store")
STORE = RESULTS / f"{STORE_NAME}.json"
STATE_ROOT = RESULTS / ("operations" if STORE_NAME == "store" else f"operations-{STORE_NAME}")


def providers_in(profile: dict) -> set[str]:
    found = set()

    def walk(o):
        if isinstance(o, dict):
            if "provider" in o:
                found.add(o["provider"])
            for v in o.values():
                walk(v)
    walk(profile)
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True, type=Path)
    ap.add_argument("--profile", required=True, type=Path)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--estimate-usd", type=float, required=True, help="expected spend of this run for its main budgeted provider")
    ap.add_argument("--timeout-s", type=float, default=3600)
    ap.add_argument("--data-dir", type=Path, default=W / "data", help="data folder mounted read-only for the run")
    a = ap.parse_args()
    wf = (PROJECT / a.workflow).resolve()
    prof_path = (PROJECT / a.profile).resolve()
    profile = json.loads(prof_path.read_text())
    provs = providers_in(profile)
    key_names = tuple(sorted({"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}[p] for p in provs))

    # 1. budget
    t = tally()
    for p in provs:
        info = t["providers"].get(p)
        if info and info["budget_usd"] is not None:
            if info["unpriced_models"]:
                raise SystemExit(f"refusing: {p} has unpriced models {info['unpriced_models']}")
            if info["spent_usd"] + a.estimate_usd > info["stop_at_usd"]:
                raise SystemExit(f"refusing: {p} ${info['spent_usd']:.2f} + estimate ${a.estimate_usd:.2f} > stop ${info['stop_at_usd']:.2f}")
            print(f"budget ok: {p} ${info['spent_usd']:.2f} spent, +${a.estimate_usd:.2f} est., stop at ${info['stop_at_usd']:.2f}")

    # 2. prepare
    RUNS.mkdir(parents=True, exist_ok=True)
    req = ReusableCandidatePreparationRequest(
        source_path=str(wf), project_root=str(PROJECT),
        expected_source_sha256=hashlib.sha256(wf.read_bytes()).hexdigest(),
        runs_root=str(RUNS), run_id=uuid4(), run_name=a.run_name,
        llm_config_file=str(prof_path),
        preparation_timeout_seconds=600.0, candidate_timeout_seconds=a.timeout_s,
        task_read_only=(str(W / "package/directed_lr_coupling"), str(a.data_dir.resolve())),
        candidate_network=True, candidate_devices=("/dev/dxg",),
        candidate_environment_names=key_names,
    )
    req_path = RESULTS / f"request-{a.run_name}.json"
    req_path.write_text(req.model_dump_json(indent=2))
    r = subprocess.run([sys.executable, "-m", "tools.candidate_prepare", "--reusable-request", str(req_path)],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("prepare failed")
    receipt = json.loads(r.stdout)
    print("prepared:", receipt["envelope"])

    # 3. submit
    if not STORE.exists():
        STORE.write_text(json.dumps({"state_root": str(STATE_ROOT),
                                     "preparation_roots": [str(RUNS)], "candidate_roots": [str(RUNS)]}, indent=2))
    cfg = OperationStoreConfig.model_validate_json(STORE.read_bytes())
    sub = build_reusable_operation(cfg, select_reusable_preparation(receipt["envelope"]), uuid4())
    sub_path = RESULTS / f"submission-{a.run_name}.json"
    sub_path.write_text(sub.model_dump_json(indent=2))
    s = subprocess.run([sys.executable, "-m", "tools.candidate_operations", "submit",
                        "--store-config", str(STORE), "--request", str(sub_path)], capture_output=True, text=True)
    if s.returncode:
        print(s.stdout[-3000:], s.stderr[-3000:])
        raise SystemExit("submit failed")
    op = json.loads(s.stdout)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_name": a.run_name, "operation_id": op["operation_id"],
           "envelope": receipt["envelope"], "workflow": str(wf), "profile": str(prof_path),
           "estimate_usd": a.estimate_usd, "interpreter": sys.executable}
    with (RESULTS / "launches.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
