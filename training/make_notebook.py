"""
Converts colab_online_grpo.py → IntelliCredit_GRPO.ipynb
Splits on  # ═══ CELL N:  markers.
Run: python training/make_notebook.py
"""
import json, re, os

SRC = os.path.join(os.path.dirname(__file__), "colab_online_grpo.py")
DST = os.path.join(os.path.dirname(__file__), "IntelliCredit_GRPO.ipynb")

raw = open(SRC, encoding="utf-8").read()

CELL1_SOURCE = [
    "# IntelliCredit Online GRPO — Cell 1: Install\n",
    "# After install completes, click 'Restart session', then run Cells 2 onwards.\n",
    "# Do NOT re-run this cell after restart.\n",
    "!pip install -U 'transformers>=4.45.0' peft accelerate 'bitsandbytes>=0.46.1' -q\n",
    "!pip install -U 'trl>=0.15.2' datasets matplotlib huggingface_hub requests -q\n",
    "print('Done. Restart session now, then run from Cell 2.')\n",
]

CELL_RE = re.compile(r"^# \u2550+\s*CELL\s+\d+.*\u2550*\s*$", re.MULTILINE)

body_start = raw.find("# \u2550\u2550\u2550 CELL 2:")
if body_start == -1:
    body_start = raw.find("# \u2550\u2550\u2550 CELL 1:")
body = raw[body_start:]

segments = CELL_RE.split(body)

cells_source = []
for seg in segments:
    if not seg.strip():
        continue
    lines = seg.splitlines(keepends=True)
    while lines and not lines[0].strip():
        lines.pop(0)
    cells_source.append(lines)

def code_cell(source_lines, cell_id):
    return {"cell_type":"code","execution_count":None,"id":cell_id,
            "metadata":{},"outputs":[],"source":source_lines}

def md_cell(text, cell_id):
    return {"cell_type":"markdown","id":cell_id,"metadata":{},"source":[text]}

nb_cells = []
nb_cells.append(md_cell(
    "# IntelliCredit Online GRPO Training\n"
    "**HOW TO USE:**\n"
    "1. Run **Cell 1** (install) -> click **Restart session** when prompted\n"
    "2. Run **Cells 2 -> 15** in order (do NOT re-run Cell 1)\n",
    "cell-md-header"
))
nb_cells.append(code_cell(CELL1_SOURCE, "cell-01-install"))

for idx, src_lines in enumerate(cells_source, start=2):
    nb_cells.append(code_cell(src_lines, f"cell-{idx:02d}"))

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": [], "name": "IntelliCredit_GRPO.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": nb_cells,
}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written: {DST} ({len(nb_cells)} cells)")
