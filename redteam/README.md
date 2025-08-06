# RedTeam Scaffold for gpt-oss-20b Hackathon

This repository contains an **automated red-teaming scaffold** designed to discover and document vulnerabilities in the open-weights model **gpt-oss-20b** across nine exploit classes.

Directory layout:

- `redteam_engine.py` – main orchestrator script (see `python redteam_engine.py --help`)
- `findings/` – auto-generated JSON files (`findings.myteam.N.json`)
- `writeup/` – markdown write-ups for each finding; combine these into your Kaggle report
- `notebook/` – lightweight notebook stub (`reproduce_findings.py`) to re-run prompts and assert outputs

> **NOTE**: By default the engine runs in *dry-run* mode and does **not** load the 20-B model. Pass `--run` (and ensure adequate GPU memory) to execute prompts live.