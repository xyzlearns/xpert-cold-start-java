"""Phase K — Output schema validation and serialization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .core_index import (
    VALID_CIRCLES,
    VALID_GOAL_AREAS,
    VALID_GOAL_TYPES,
    VALID_HELP_ACTIONS,
    VALID_HORIZONS,
    VALID_LINK_TYPES,
    VALID_LOOP_TYPES,
    VALID_ORG_ROLES,
    VALID_ORG_SIDES,
    VALID_ORG_TYPES,
    VALID_STATUSES,
    ConversationIndex,
)


def validate_output(out: Dict[str, Any], idx: ConversationIndex) -> List[str]:
    """Validate output dictionary against schema.json and input IDs."""
    errs: List[str] = []
    if not isinstance(out, dict):
        return ["The output must be a JSON object."]

    required_keys = ("people", "organisations", "chats", "goals", "open_loops")
    for k in required_keys:
        if not isinstance(out.get(k), list):
            errs.append(f'"{k}" must be a list.')
    if errs:
        return errs

    extra = set(out) - set(required_keys)
    if extra:
        errs.append(f"Unknown top-level keys: {sorted(extra)}.")

    chat_ids = set(idx.chats_by_id.keys())
    msg_ids = set(idx.messages_by_id.keys())
    person_ids = set(idx.participant_ids)
    org_ids = set()

    for i, o in enumerate(out["organisations"]):
        where = f"organisations[{i}]"
        if not isinstance(o, dict) or not o.get("id"):
            errs.append(f"{where} needs an id.")
            continue
        if o["id"] in org_ids:
            errs.append(f'{where}: the id "{o["id"]}" is used twice.')
        org_ids.add(o["id"])
        if not o.get("name"):
            errs.append(f"{where} needs a name.")
        if o.get("type") not in VALID_ORG_TYPES:
            errs.append(f"{where}: invalid type {o.get('type')}.")
        if o.get("side") not in VALID_ORG_SIDES:
            errs.append(f"{where}: invalid side {o.get('side')}.")
        if o.get("role") not in VALID_ORG_ROLES:
            errs.append(f"{where}: invalid role {o.get('role')}.")

    seen_people = set()
    for i, p in enumerate(out["people"]):
        where = f"people[{i}]"
        pid = p.get("id") if isinstance(p, dict) else None
        if pid not in person_ids:
            errs.append(f'{where}: "{pid}" is not a valid person in input.')
            continue
        if pid in seen_people:
            errs.append(f'{where}: "{pid}" is listed twice.')
        seen_people.add(pid)
        if p.get("circle") not in VALID_CIRCLES:
            errs.append(f"{where}: invalid circle {p.get('circle')}.")
        for link in p.get("links", []) or []:
            if link.get("organisation") not in org_ids:
                errs.append(f'{where}: unknown organisation "{link.get("organisation")}".')
            if link.get("link") not in VALID_LINK_TYPES:
                errs.append(f"{where}: invalid link {link.get('link')}.")
        for other in p.get("looks_after", []) or []:
            if other not in person_ids:
                errs.append(f'{where}: unknown looks_after person "{other}".')

    tied = set()
    for i, c in enumerate(out["chats"]):
        where = f"chats[{i}]"
        cid = c.get("chat_id") if isinstance(c, dict) else None
        if cid not in chat_ids:
            errs.append(f'{where}: "{cid}" is not a chat in the input.')
            continue
        if cid in tied:
            errs.append(f'{where}: "{cid}" is tied twice.')
        tied.add(cid)
        has_p, has_o = "person" in c, "organisation" in c
        if has_p == has_o:
            errs.append(f"{where}: give exactly one of person or organisation.")
        elif has_p and c["person"] not in person_ids:
            errs.append(f'{where}: unknown person "{c["person"]}".')
        elif has_o and c["organisation"] not in org_ids:
            errs.append(f'{where}: unknown organisation "{c["organisation"]}".')
    missing = chat_ids - tied
    if missing:
        errs.append(f"{len(missing)} chat(s) are not tied to anyone.")

    if len(out["goals"]) > 10:
        errs.append("At most 10 goals.")
    for i, g in enumerate(out["goals"]):
        where = f"goals[{i}]"
        if not g.get("title"):
            errs.append(f"{where} needs a title.")
        if g.get("area") not in VALID_GOAL_AREAS:
            errs.append(f"{where}: invalid area.")
        if g.get("type") not in VALID_GOAL_TYPES:
            errs.append(f"{where}: invalid type.")
        if g.get("horizon") not in VALID_HORIZONS:
            errs.append(f"{where}: invalid horizon.")
        ev = g.get("evidence") or []
        if not ev:
            errs.append(f"{where}: give at least one evidence message.")
        for m in ev:
            if m not in msg_ids:
                errs.append(f'{where}: evidence "{m}" is not in input messages.')

    for i, l in enumerate(out["open_loops"]):
        where = f"open_loops[{i}]"
        if not l.get("title"):
            errs.append(f"{where} needs a title.")
        if l.get("type") not in VALID_LOOP_TYPES:
            errs.append(f"{where}: invalid type.")
        if l.get("owner") != "ME" and l.get("owner") not in person_ids:
            errs.append(f"{where}: invalid owner.")
        if l.get("chat_id") not in chat_ids:
            errs.append(f"{where}: unknown chat_id.")
        if l.get("source_msg") not in msg_ids:
            errs.append(f"{where}: unknown source_msg.")
        if l.get("status") not in VALID_STATUSES:
            errs.append(f"{where}: invalid status.")
        if l.get("closed_by") is not None and l.get("closed_by") not in msg_ids:
            errs.append(f"{where}: unknown closed_by.")
        help_ = l.get("help")
        if help_ is not None:
            if l.get("status") != "open":
                errs.append(f"{where}: help is only for open loops.")
            if not isinstance(help_, dict) or help_.get("action") not in VALID_HELP_ACTIONS:
                errs.append(f"{where}: invalid help.action.")

    return errs


def write_output(out: Dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
