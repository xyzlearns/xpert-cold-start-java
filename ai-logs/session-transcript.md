# AI Engineering Session Transcript — Xpert Cold Start Challenge

## Session 1: Repository & Schema Inspection (Phase A & B)
- **Objective:** Inspect `schema.json`, `check.py`, `solution.py`, `llm.py`, `run.sh`, `example/`, and `data/dev/`.
- **Findings:**
  - `check.py` validates exact JSON shape (`people`, `organisations`, `chats`, `goals`, `open_loops`) with `additionalProperties: false` and scores 74 points on `example/` and 56 points on `data/dev/` (where `goals` and `help` are unlabelled).
  - Message timestamps (`ts`) require explicit chronological sorting (`(ts, line_no, id)`) because message IDs (`M001`, `M026`, etc.) are not strictly ordered by timestamp.
- **Implementation:** Created `src/xpert/ingestion.py` and `tests/test_ingestion.py`. Verified 100% pass on both `example/` (6 chats, 29 messages, 8 participants) and `data/dev/` (138 chats, 4,502 messages, 286 participants).

## Session 2: Entity, Organisation, Chat & Goal Extraction (Phase C, D, E, F)
- **Objective:** Classify people into relationship circles, cluster chats into organisations, tie every chat exclusively to one person or organisation, and extract up to 10 evidenced goals.
- **Key Decisions:**
  - Preserved all `seed.json` circles and organisations without overwriting.
  - Built hierarchical relationship classification (`spouse`, `child`, `staff`, `parent`, `sibling`, `in-law`, `relative`, `partner`, `colleague`, `client`, `vendor`, `friend`) using contact naming patterns, group co-membership, and conversational honorifics.
  - Optimised organisation-to-chat and organisation-to-person links (`runs` vs. `supports`) to align with `check.py`'s bipartite Jaccard matcher (`match_orgs`).

## Session 3: High-Precision Open-Loop Detection, Resolution & Error Analysis (Phase G, H, I, J)
- **Objective:** Maximize open-loop detection F1, status accuracy, and help-action accuracy while respecting `check.py`'s `-1` penalty for false closures and `-1` penalty for intrusive help on `leave` loops.
- **Error Analysis & Refinements:**
  - Filtered out automated broadcast/news feeds (`mediadomain4`, `Nexus Meta AI`) that generated false-positive open loops.
  - Fixed a string-matching bug where `Nayna Khetan` (spouse) was caught by a substring check on `"Never Trust Anyone 💯❤️💯"` due to the shared `❤️` symbol, lifting Dev Circles score from `0.71` (`8.5/12`) to `1.00` (`12.0/12`).
  - Refined chronological thread resolution for commitments, follow-ups, questions, and deadlines across 1:1 and operational group chats, lifting Dev total score from `5.4 / 56` (starter baseline) to `53.1 / 56` (`94.8%` of available dev points) and `74.0 / 74` (`100%`) on `example/`.

## Session 4: Packaging & Verification (Phase K & L)
- **Objective:** Package clean submission with `run.sh`, `solution.py`, `src/xpert/`, `tests/`, `WRITEUP.md`, `requirements.txt`, and `.env.example`.
- **Verification:** Ran `./run.sh` and `check.py` on both `example` and `data/dev`, confirming zero schema errors, sub-second execution, and deterministic output.
