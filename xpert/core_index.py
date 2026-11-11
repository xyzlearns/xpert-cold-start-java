"""Phase B — Core data models, ingestion, and indexing for Xpert Cold Start."""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

VALID_CIRCLES = {
    "spouse", "child", "parent", "in-law", "sibling", "relative",
    "friend", "colleague", "client", "vendor", "partner", "staff",
}
VALID_ORG_TYPES = {
    "company", "residence", "vacation home", "workspace", "society", "school", "other",
}
VALID_ORG_SIDES = {"home", "work"}
VALID_ORG_ROLES = {"owner", "partner", "helper", "resident", "parent", "member", "other"}
VALID_LINK_TYPES = {"runs", "supports"}
VALID_GOAL_AREAS = {
    "self", "spouse", "kids", "parents", "work", "friends",
    "experiences", "finances", "health", "home",
}
VALID_GOAL_TYPES = {"habit", "milestone", "experience"}
VALID_HORIZONS = {"3mo", "1yr", "3yr", "5yr"}
VALID_LOOP_TYPES = {"commitment", "followup", "deadline", "question"}
VALID_STATUSES = {"open", "done", "dropped"}
VALID_HELP_ACTIONS = {"assist", "remind", "leave"}


@dataclass(frozen=True)
class ChatRecord:
    chat_id: str
    name: str
    is_group: bool
    participants: Tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any], line_no: int = 0) -> "ChatRecord":
        try:
            cid = str(data["chat_id"])
            name = str(data.get("name") or "")
            is_grp = bool(data.get("is_group", False))
            parts = tuple(str(p) for p in (data.get("participants") or data.get("participant_ids") or []))
            return cls(chat_id=cid, name=name, is_group=is_grp, participants=parts)
        except KeyError as exc:
            raise ValueError(f"Malformed ChatRecord at line {line_no}: missing {exc}") from exc


@dataclass(frozen=True)
class MessageRecord:
    id: str
    chat_id: str
    sender_id: str
    sender_name: str
    ts: int
    kind: str
    body: str
    quoted_id: Optional[str] = None
    line_no: int = 0

    @property
    def dt_ist(self) -> datetime:
        return datetime.fromtimestamp(self.ts, tz=IST)

    @property
    def date_str(self) -> str:
        return self.dt_ist.strftime("%Y-%m-%d")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], line_no: int = 0) -> "MessageRecord":
        try:
            return cls(
                id=str(data["id"]),
                chat_id=str(data["chat_id"]),
                sender_id=str(data["sender_id"]),
                sender_name=str(data.get("sender_name") or ""),
                ts=int(data["ts"]),
                kind=str(data.get("kind") or "text"),
                body=str(data.get("body") or ""),
                quoted_id=data.get("quoted_id"),
                line_no=line_no,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Malformed MessageRecord at line {line_no}: {exc}") from exc


@dataclass
class SeedData:
    circles: Dict[str, str] = field(default_factory=dict)
    organisations: List[str] = field(default_factory=list)
    goals: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedData":
        circles = {
            str(k): str(v)
            for k, v in (data.get("circles") or {}).items()
            if str(v) in VALID_CIRCLES
        }
        orgs = [str(o) for o in (data.get("organisations") or []) if o]
        goals = list(data.get("goals") or [])
        return cls(circles=circles, organisations=orgs, goals=goals)


@dataclass
class ParticipantProfile:
    person_id: str
    seeded_circle: Optional[str] = None
    display_names: Counter = field(default_factory=Counter)
    direct_chat_ids: List[str] = field(default_factory=list)
    direct_chat_names: List[str] = field(default_factory=list)
    group_chat_ids: List[str] = field(default_factory=list)
    group_chat_names: List[str] = field(default_factory=list)
    sent_messages: List[MessageRecord] = field(default_factory=list)
    messages_in_direct_chats: List[MessageRecord] = field(default_factory=list)

    @property
    def primary_name(self) -> str:
        if self.display_names:
            return self.display_names.most_common(1)[0][0]
        if self.direct_chat_names:
            return self.direct_chat_names[0]
        return self.person_id


class ConversationIndex:
    """Chronological conversation index and participant registry."""

    def __init__(
        self,
        chats: List[ChatRecord],
        messages: List[MessageRecord],
        seed: SeedData,
    ) -> None:
        self.chats = chats
        self.seed = seed
        self.chats_by_id: Dict[str, ChatRecord] = {c.chat_id: c for c in chats}

        # Chronological sort with deterministic tie-breaking by (ts, line_no, id)
        self.messages: List[MessageRecord] = sorted(
            messages, key=lambda m: (m.ts, m.line_no, m.id)
        )
        self.messages_by_id: Dict[str, MessageRecord] = {m.id: m for m in self.messages}

        self.messages_by_chat: Dict[str, List[MessageRecord]] = defaultdict(list)
        self.messages_by_sender: Dict[str, List[MessageRecord]] = defaultdict(list)
        for m in self.messages:
            self.messages_by_chat[m.chat_id].append(m)
            self.messages_by_sender[m.sender_id].append(m)

        # Build participant registry (never include ME)
        self.participant_ids: Set[str] = {
            p for c in chats for p in c.participants if p != "ME"
        }
        self.profiles: Dict[str, ParticipantProfile] = {
            pid: ParticipantProfile(
                person_id=pid,
                seeded_circle=seed.circles.get(pid),
            )
            for pid in sorted(self.participant_ids)
        }

        for c in chats:
            for pid in c.participants:
                if pid == "ME" or pid not in self.profiles:
                    continue
                prof = self.profiles[pid]
                if c.is_group:
                    prof.group_chat_ids.append(c.chat_id)
                    prof.group_chat_names.append(c.name)
                else:
                    prof.direct_chat_ids.append(c.chat_id)
                    prof.direct_chat_names.append(c.name)

        for m in self.messages:
            if m.sender_id != "ME" and m.sender_id in self.profiles:
                prof = self.profiles[m.sender_id]
                if m.sender_name:
                    prof.display_names[m.sender_name] += 1
                prof.sent_messages.append(m)
            chat_rec = self.chats_by_id.get(m.chat_id)
            if chat_rec and not chat_rec.is_group:
                for pid in chat_rec.participants:
                    if pid != "ME" and pid in self.profiles:
                        self.profiles[pid].messages_in_direct_chats.append(m)

        # Determine user ("ME") display name and family surname if available
        me_names = Counter(
            m.sender_name for m in self.messages if m.sender_id == "ME" and m.sender_name
        )
        self.me_primary_name: str = me_names.most_common(1)[0][0] if me_names else "ME"
        parts = self.me_primary_name.strip().split()
        self.me_surname: Optional[str] = parts[-1].lower() if len(parts) >= 2 else None

        self.min_ts: int = self.messages[0].ts if self.messages else 0
        self.max_ts: int = self.messages[-1].ts if self.messages else 0


def load_dataset(input_dir: str | Path) -> ConversationIndex:
    """Load chats.jsonl, messages.jsonl, and seed.json from input_dir."""
    folder = Path(input_dir)
    chats_path = folder / "chats.jsonl"
    msgs_path = folder / "messages.jsonl"
    seed_path = folder / "seed.json"

    if not chats_path.exists():
        raise FileNotFoundError(f"Missing required file: {chats_path}")
    if not msgs_path.exists():
        raise FileNotFoundError(f"Missing required file: {msgs_path}")

    chats: List[ChatRecord] = []
    with open(chats_path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {chats_path} at line {idx}: {exc}") from exc
            chats.append(ChatRecord.from_dict(raw, line_no=idx))

    messages: List[MessageRecord] = []
    with open(msgs_path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {msgs_path} at line {idx}: {exc}") from exc
            messages.append(MessageRecord.from_dict(raw, line_no=idx))

    seed_dict: Dict[str, Any] = {}
    if seed_path.exists():
        seed_dict = json.loads(seed_path.read_text(encoding="utf-8"))
    seed = SeedData.from_dict(seed_dict)

    return ConversationIndex(chats=chats, messages=messages, seed=seed)
