"""Generate a research-loop workflow.py for cell-signalling from the proven template.

    python project/scripts/make_workflow.py --name smoke --iterations 1 --rounds 1 [--advice-file advice/decode_findings.md]

Template: the research-loop declaration created by the SIDERIUS console
(Interpret -> Propose -> Implement -> Validate -> Train & tune, feedback to
Interpret), which ran end to end on this laptop on 2026-11-20. Only the task
paths, iteration count, tuning rounds and advice are changed. The advice text
is the TEAM's file, passed through verbatim as proposal advice.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

W = Path.home() / "work/cell-signalling"
PROJECT = W / "project"
TEMPLATE = Path.home() / "siderius-project/siderius_workspace/projects/quickstart-research-loop/workflow.py"
MANIFEST = W / "package/directed_lr_coupling/task/task.yaml"
DATA = W / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=2, help="tuning rounds per iteration (last is Formal)")
    ap.add_argument("--advice-file", type=Path)
    ap.add_argument("--data-root", type=Path, default=DATA, help="data folder (train.npz + validation.npz)")
    a = ap.parse_args()

    src = TEMPLATE.read_text()
    src, n1 = re.subn(r"manifest='[^']*'", f"manifest='{MANIFEST}'", src)
    src, n2 = re.subn(r"data_root='[^']*'", f"data_root='{a.data_root.resolve()}'", src)
    src, n3 = re.subn(r"final_iteration=\d+", f"final_iteration={a.iterations}", src)
    src, n4 = re.subn(r"'max_rounds': \d+", f"'max_rounds': {a.rounds}", src)
    # The task's Model-I/O contract is a continuous [B, 1] score (no class axis); SIDERIUS refuses a
    # 'classifier' plugin against it (smoke run 2026-11-21), so only 'regressor' plugins are allowed.
    src, n6 = re.subn(r"allowed_output_types=None", "allowed_output_types=('regressor',)", src)
    if (n1, n2, n3, n4, n6) != (1, 1, 1, 1, 1):
        raise SystemExit(f"template substitution failed: {(n1, n2, n3, n4, n6)}")
    if a.advice_file:
        advice = a.advice_file.read_text().strip()
        if not advice:
            raise SystemExit(f"{a.advice_file} is empty")
        # A function replacement keeps backslash escapes (e.g. \n in the repr) literal.
        src, n5 = re.subn(r"proposal_advice=None", lambda _m: f"proposal_advice={advice!r}", src)
        if n5 != 1:
            raise SystemExit("could not set proposal_advice")
    out = PROJECT / "experiment" / f"workflow-{a.name}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    print(f"wrote {out} (iterations={a.iterations}, rounds={a.rounds}, advice={'yes' if a.advice_file else 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
