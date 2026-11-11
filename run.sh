#!/usr/bin/env bash
# The one command we run:  ./run.sh <input_dir> <output.json>
# Change what it calls, not how it is called. It must finish in 30 minutes.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: ./run.sh <input_dir> <output.json>" >&2
  exit 2
fi
project_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
build_dir="$project_dir/.build"
mkdir -p "$build_dir"
javac --release 17 -encoding UTF-8 -d "$build_dir" "$project_dir/src/main/java/ColdStartSolution.java"
java -cp "$build_dir" ColdStartSolution "$1" "$2"
