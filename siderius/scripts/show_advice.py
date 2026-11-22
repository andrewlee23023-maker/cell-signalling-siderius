"""Print the exact proposal advice a generated workflow passes to the proposal agent,
and confirm it is byte-identical to the team's advice file.

    python3 show_advice.py WORKFLOW.py [ADVICE_FILE]
"""
import ast
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
found = {}
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.keyword) and node.arg in ("proposal_advice", "allowed_output_types"):
        found[node.arg] = ast.literal_eval(node.value)
for k, v in found.items():
    print(f"{k} =\n{v}\n")
if len(sys.argv) > 2:
    team = Path(sys.argv[2]).read_text().strip()
    print("identical to the team's advice file:", found.get("proposal_advice") == team)
