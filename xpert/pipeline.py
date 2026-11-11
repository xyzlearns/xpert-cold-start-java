"""Orchestrate deterministic extraction and optional, bounded LLM adjudication."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from .core_index import (
    ChatRecord,
    ConversationIndex as CoreConversationIndex,
    MessageRecord,
    SeedData as CoreSeedData,
    load_dataset,
)
from .entities import extract_organisations_and_chats, extract_people
from .goals import extract_goals
from .indexes import ConversationIndex, ParticipantRegistry
from .llm_client import LLMClient
from .models import InputData
from .open_loops import extract_open_loops
from .validator import validate_output, write_output

logger = logging.getLogger(__name__)


def _to_core_index(data: InputData) -> CoreConversationIndex:
    """Convert InputData into CoreConversationIndex for unified extraction."""
    chats = [
        ChatRecord(
            chat_id=c.chat_id,
            name=c.name,
            is_group=c.is_group,
            participants=tuple(c.participants),
        )
        for c in data.chats
    ]
    messages = [
        MessageRecord(
            id=m.id,
            chat_id=m.chat_id,
            sender_id=m.sender_id,
            sender_name=m.sender_name or "",
            ts=int(m.ts),
            kind=m.kind,
            body=m.body,
            quoted_id=m.quoted_id,
            line_no=idx_pos,
        )
        for idx_pos, m in enumerate(data.messages, start=1)
    ]
    seed = CoreSeedData(
        circles=dict(data.seed.circles),
        organisations=list(data.seed.organisations),
        goals=[dict(g) for g in data.seed.goals],
    )
    return CoreConversationIndex(chats=chats, messages=messages, seed=seed)


def build_baseline(
    data: InputData,
    registry: ParticipantRegistry | None = None,
    index: ConversationIndex | None = None,
) -> Dict[str, Any]:
    """Produce a schema-validated, high-precision extraction result."""
    core_idx = _to_core_index(data)
    llm = LLMClient(cache_dir=Path(".cache") / "llm")

    organisations, chat_to_org, chats_out = extract_organisations_and_chats(core_idx, llm=llm)
    people_out = extract_people(core_idx, organisations, chat_to_org, llm=llm)
    goals_out = extract_goals(core_idx, llm=llm)
    open_loops_out = extract_open_loops(core_idx, people_out, llm=llm)

    out: Dict[str, Any] = {
        "people": people_out,
        "organisations": organisations,
        "chats": chats_out,
        "goals": goals_out,
        "open_loops": open_loops_out,
    }

    errs = validate_output(out, core_idx)
    if errs:
        logger.warning("Output validation found %d issue(s): %s", len(errs), errs[:5])

    return out


def run_pipeline(input_dir: str | Path, output_path: str | Path) -> Dict[str, Any]:
    """Execute the end-to-end Cold Start extraction pipeline and write output.json."""
    core_idx = load_dataset(input_dir)
    llm = LLMClient(cache_dir=Path(".cache") / "llm")

    organisations, chat_to_org, chats_out = extract_organisations_and_chats(core_idx, llm=llm)
    people_out = extract_people(core_idx, organisations, chat_to_org, llm=llm)
    goals_out = extract_goals(core_idx, llm=llm)
    open_loops_out = extract_open_loops(core_idx, people_out, llm=llm)

    out: Dict[str, Any] = {
        "people": people_out,
        "organisations": organisations,
        "chats": chats_out,
        "goals": goals_out,
        "open_loops": open_loops_out,
    }

    errs = validate_output(out, core_idx)
    if errs:
        logger.warning("Output validation found %d issue(s): %s", len(errs), errs[:5])

    write_output(out, output_path)
    return out
