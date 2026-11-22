# Run declaration — Track 2 · cell-signalling

Phase: preparation; execution is not yet configured or authorized.
Project: /home/andre/work/cell-signalling/project
Task instructions and manifest: UNCONFIGURED until the noon release.
  Package will be unzipped under /home/andre/work/cell-signalling/package; read its TASK_CARD.md and baseline first.
Experiment and allowed data/partitions: UNCONFIGURED until the noon release.
  Data will be fetched to /home/andre/work/cell-signalling/data. Train split only for fitting; validation only for
  comparison and configuration selection; never test. Split by tissue section if a
  section ID exists (see ../decode_report.md).
Enabled capabilities and model routing: UNCONFIGURED.
  Profiles available in this project: config/llm-smoke.json (all Gemini 3.1 Flash-Lite,
  smoke tests only) and config/llm-moderate.json (Claude Sonnet 5.5 everywhere except
  validation on Gemini 3.1 Flash-Lite; SIDERIUS v1.1.0 accepts only gemini/openai/deepseek
  for validation). Skip or minimize literature review.
Shared API/training/wall-time limits: SET by the team on 2026-11-21.
  API: Anthropic $25.00, OpenAI $30.00 (key pending) — config/budget.json. The caller must call
  scripts/spend_guard.py check_budget() before EVERY capability invocation; it stops new calls at
  90% of a provider's budget ($22.50 / $27.00), pricing every prompt token at the input rate
  (over-estimate). Gemini (validation only) is tracked in tokens, not capped.
  Provider-console spend limits are the hard backstop; set them too.
  Compute: this laptop (RTX 5060 8 GB, WSL); per-invocation 3,600 s (sandbox.json).
  Wall time: loop iterations 13:00-15:00 PST; freeze by 15:00 PST; models close 16:00 PST.
  Planning figure: one Sonnet 5.5 loop iteration cost about $0.50 in yesterday's test runs.
Authorized evidence and advice files: UNCONFIGURED.
  Intended: advice/decode_findings.md, written by the team from ../decode_report.md.
Caller script and required result files: UNCONFIGURED (scripts/caller.py, to be authored).
Qualified framework revision/environment: SIDERIUS v1.1.0 (official checkout ~/SIDERIUS)
  for anything submitted. Local WSL GPU runs need ~/SIDERIUS-wsl (local GPU-accounting
  patch); disclose it if any reported evidence comes from that checkout.

During preparation, author task and experiment files in this project.
Before execution, finish these declarations and protect selected inputs
through sandbox.json. Installing these instructions grants no run budget.
