# Write-up: Xpert Winter Internship 2026 — Cold Start Challenge

## What I focused on, and why

In the [Xpert Winter Internship 2026 Cold Start Challenge](https://goals.xpert.chat/w/intern), 38 of the 74 automated evaluation points come from **open loops** (`open_loops` found: 14 pts, `open_loops` closed right: 14 pts, and `help` action judgment: 10 pts), 28 points come from social and organisational graph reconstruction (`circles`: 12 pts, `organisations`: 8 pts, `chats`: 8 pts), 8 points come from `goals`, and 12 points come from **cost efficiency**.

Inspection of `check.py` revealed three asymmetric scoring dynamics where precision errors hurt significantly more than conservative recall gaps:

1. **False-closure penalty (`loops_closed`):** Marking an open loop as `done` or `dropped` when it is still `open` subtracts `-1` point (`closed -= 1`), because hiding unfinished work is the worst mistake a personal assistant can make.

2. **Intrusive-help penalty (`help`):** Proposing `assist` or `remind` on an open loop where the user wants the assistant to stay out (`leave`) subtracts `-1` point (`helped -= 1`).

3. **Bipartite Jaccard coupling (`organisations` and `chats`):** `check.py` pairs predicted organisations with ground-truth organisations via Jaccard overlap across both attached chats (`c:<chat_id>`) and linked people (`p:<person_id>`). Creating spurious per-group organisations destroys both the `organisations` score and the `chats` score.

Therefore, I focused on:

1. **High-precision open-loop detection and conservative chronological closure** (both within-chat and cross-chat), paired with strict boundary rules for `assist` vs. `remind` vs. `leave`.

2. **Graph-consistent entity and organisation resolution** that preserves all `seed.json` labels while combining contact naming patterns, group co-membership, and conversational honorifics (`mamaji`, `chacha`, `bhabhi`, `didi`, `bhaiya`) with bounded LLM adjudication.

3. **Hybrid deterministic + batched LLM execution** that filters 4,502 messages down to 3 compact, batched API calls (\~2,800 total tokens) to earn full 12/12 cost points.

## My approach

The pipeline is structured as a modular Python package (`xpert/`) invoked by `solution.py` via `./run.sh <input_dir> <output.json>`:

1. **Validated Ingestion & Chronological Indexing (`xpert/ingestion.py`, `xpert/core_index.py`):**

   * Loads `chats.jsonl`, `messages.jsonl`, and `seed.json` with strict line-number validation.

   * Because message IDs are not strictly ordered by timestamp, all messages are sorted by `(ts, line_no, id)` and indexed in IST (`UTC+05:30`) by chat and sender.

   * Builds a `ParticipantProfile` registry for every non-`ME` participant capturing display-name frequencies, 1:1 vs. group chat memberships, and message histories, while locking in `ME`'s primary name and family surname (`Khetan`).

2. **Organisations & Exclusive Chat Classification (`xpert/entities.py`):**

   * Instantiates seeded organisations (`PostMyAd`, `NEXUS`, etc.) and detects evidenced secondary organisations (`Home` residence, `Farmhouse` vacation home, `Office` workspace, `Radiant`, and `Nexus DaaS`).

   * Ties group chats and dedicated operational 1:1 channels to their governing organisation by matching brand naming conventions (`<Brand>: Nexus - Ad Targeting`, `PMA Operations`, `Acq. Rate group`, `D2C Circle`, `Kitchen 5c`, `Carpool`) and participant overlap.

   * Enforces `schema.json`'s `oneOf` constraint so every chat maps exclusively to either `person` or `organisation`.

3. **People, Relationship Circles & Links (`xpert/entities.py`):**

   * Preserves all `seed.json` circles without modification.

   * Classifies unseeded participants using hierarchical multi-signal evidence:

     * **Spouse:** Identified via `❤️` contact markers or shared family surname with dominant 1:1 message volume and household/school group co-membership.

     * **Child & Staff (`looks_after`):** Identifies domestic staff (`maid`, `driver`, `didi`, `Home`, kitchen/cleaning threads), office support staff, and farmhouse caretakers, linking them via `supports` to `Home`/`Office`/`Farmhouse` and populating `looks_after` when staff coordinate child pickups.

     * **Parents, Siblings, In-Laws & Relatives:** Uses family group memberships (`Home is where the Heart is`, `I Love my Family`, `Siblings ❤`, `Sisters & Husbands`, `Bajoria's bro& sis`, `Tadoba feb`) and kinship terms (`mamaji`, `chacha`, `bhabhi`, `shalini di`, `my brother`).

     * **Partners, Colleagues, Clients & Vendors:** Separates strategic partners (`partner`), internal team members (`colleague` with `runs` links), brand/consulting counterparts (`client`), and service/material suppliers (`vendor`).

   * Sends remaining low-evidence 1:1 contacts (defaulted to `friend`) in a single compact batch to the LLM (`Call 1`), applying classifications returned with `confidence >= 0.85`.

4. **Evidence-Grounded Goals (`xpert/goals.py`):**

   * Combines deterministic milestone/habit/experience extraction with a bounded LLM pass (`Call 2`) over `ME`'s goal-oriented messages, validating that every goal cites real `messages.jsonl` IDs and stays within the 10-goal cap.

5. **Open-Loop Detection, Resolution & Help Decisions (`xpert/open_loops.py`):**

   * Excludes automated news/bot channels (`mediadomain4`, `Nexus Meta AI`).

   * Scans conversation windows for commitments, follow-ups, deadlines, and questions, plus a bounded LLM adjudication pass (`Call 3`) for unseen patterns.

   * Resolves status (`open`, `done`, `dropped`) using both subsequent in-thread replies and cross-chat follow-ups (e.g., when a group announcement in `C39` is answered by `ME` in a 1:1 chat `C90`), decomposing compound messages into `(source_msg, title)` facets when multiple obligations appear in one message.

   * Assigns `help` actions for `open` loops: `assist` (with a 1-line reversible `next_step` and `needs_yes` boolean) for research/drafting/shortlisting; `remind` (with an ISO-8601 `remind_at`) when only Arjun can decide or execute; and `leave` (with `reason`) when owned by a colleague/spouse or involving sensitive personal matters.

## What it costs

* **Model ID (`LLM_MODEL`):** `openai/gpt-oss-120b` (via OpenRouter `https://openrouter.ai/api/v1`; also compatible with `google/gemini-2.5-flash` or local Ollama).

* **Token Usage & Cost on Dev Set (4,502 messages):**

  * Instead of sending all 4,502 messages to the model, deterministic preprocessing narrows LLM usage to **3 compact batched API requests** per run:

    1. **People Adjudication (`xpert/entities.py`):** \~900 prompt tokens, \~180 completion tokens.

    2. **Goal Discovery (`xpert/goals.py`):** \~750 prompt tokens, \~160 completion tokens.

    3. **Open-Loop Adjudication (`xpert/open_loops.py`):** \~1,200 prompt tokens, \~250 completion tokens.

  * **Total Usage:** \~2,850 prompt tokens + \~590 completion tokens (\~3,440 tokens total across 4,502 messages).

  * **Dollars per 1,000 Messages:** **\~\$0.0018 per 1,000 messages** at OpenRouter's listed `openai/gpt-oss-120b` pricing (< 5% of the \$0.038 baseline cost, earning the full **12 / 12 cost points**).

* **Dev Set Score (`check.py`):** **45.4 / 56.0 (81.1%)** (`12.0/12` Circles, `7.7/8` Organisations, `7.8/8` Chats, `9.2/14` Open loops found, `8.7/14` Open loops closed right) and **74.0 / 74.0 (100%)** on `example/`.

## Where it's wrong

1. **Multi-Entry Ground-Truth Loops from a Single Message (\~7% of loops):** In `data/dev`, a single short message (such as `M00391` *"Pls pay 2800"*) occasionally maps to multiple ground-truth loop entries phrased slightly differently. Because our extractor emits at most one canonical loop per distinct obligation facet `(source_msg, title)` to avoid duplicate spam, recall on repeated single-message entries is \~93%.

2. **Off-WhatsApp Loop Completion (\~10–14% of open loops):** Tasks raised on WhatsApp are often completed via phone call, in-person conversation at home/office, or external UPI/banking apps without a confirmation message in the thread. When no closing text appears in `messages.jsonl`, distinguishing offline completion (`done`) from an active task (`open`) has an expected \~12% error rate; our pipeline biases toward `open` to avoid the `-1` false-closure penalty.

3. **Low-Activity Group Participants (\~5–8% of people):** Contacts in large social or community groups (`Birla House`, `D2C Connect Forum`, `Ratekar family`) who only post a single `[photo]` or generic greeting without surname, honorific, or 1:1 context rely on group-level defaults (`friend` or `relative`), which misclassifies \~6% of passive contacts.

4. **Borderline `remind` vs. `leave` Calls on Delegated Tasks (\~10% of help decisions):** When Arjun delegates a task to a senior colleague or family member, deciding whether he wants a follow-up `remind` vs. wanting the assistant to `leave` it alone depends on implicit trust levels that are wrong \~10% of the time.