#!/usr/bin/env python3
"""Run the deterministic Cold Start pipeline.

    python3 solution.py <input_dir> <output.json>
"""
import json
import sys
from pathlib import Path

from xpert.indexes import ConversationIndex, ParticipantRegistry
from xpert.ingestion import InputError, load_input
from xpert.pipeline import build_baseline

def main(input_dir, output_path):
    data = load_input(input_dir)
    registry = ParticipantRegistry.build(data)
    index = ConversationIndex.build(data)
    out = build_baseline(data, registry, index)
    Path(output_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        main(sys.argv[1], sys.argv[2])
    except InputError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        sys.exit(2)
