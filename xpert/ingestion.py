"""Validated JSONL loading; errors identify the file and record that needs correction."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Chat, InputData, Message, SeedData


class InputError(ValueError):
    """Raised when challenge input does not match the documented shape."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError(f"Could not read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise InputError(f"{path}:{line_number}: each JSONL record must be an object")
        records.append(value)
    return records


def _string(record: dict[str, Any], field: str, location: str, *, allow_null: bool = False, allow_empty: bool = False) -> str | None:
    value = record.get(field)
    if value is None and allow_null:
        return None
    if not isinstance(value, str) or (not value and not allow_empty):
        raise InputError(f"{location}: '{field}' must be a {'string' if allow_empty else 'non-empty string'}")
    return value


def _load_chats(path: Path) -> tuple[Chat, ...]:
    chats: list[Chat] = []
    ids: set[str] = set()
    for number, record in enumerate(_read_jsonl(path), start=1):
        location = f"{path}:{number}"
        chat_id = _string(record, "chat_id", location)
        name = _string(record, "name", location)
        participants = record.get("participants")
        if not isinstance(record.get("is_group"), bool):
            raise InputError(f"{location}: 'is_group' must be a boolean")
        if not isinstance(participants, list) or not participants or not all(isinstance(p, str) and p for p in participants):
            raise InputError(f"{location}: 'participants' must be a non-empty list of IDs")
        if "ME" not in participants:
            raise InputError(f"{location}: participants must include ME")
        if chat_id in ids:
            raise InputError(f"{location}: duplicate chat_id '{chat_id}'")
        ids.add(chat_id)
        chats.append(Chat(chat_id, name, record["is_group"], tuple(participants)))
    return tuple(chats)


def _load_messages(path: Path, chat_ids: set[str]) -> tuple[Message, ...]:
    messages: list[Message] = []
    ids: set[str] = set()
    kinds = {"text", "voice_transcript", "image_caption", "document"}
    for number, record in enumerate(_read_jsonl(path), start=1):
        location = f"{path}:{number}"
        message_id = _string(record, "id", location)
        chat_id = _string(record, "chat_id", location)
        sender_id = _string(record, "sender_id", location)
        sender_name = _string(record, "sender_name", location, allow_empty=True)
        body = record.get("body")
        ts = record.get("ts")
        if chat_id not in chat_ids:
            raise InputError(f"{location}: unknown chat_id '{chat_id}'")
        if message_id in ids:
            raise InputError(f"{location}: duplicate message id '{message_id}'")
        if record.get("kind") not in kinds:
            raise InputError(f"{location}: unknown message kind '{record.get('kind')}'")
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            raise InputError(f"{location}: 'ts' must be a Unix timestamp")
        if not isinstance(body, str):
            raise InputError(f"{location}: 'body' must be a string")
        quoted_id = _string(record, "quoted_id", location, allow_null=True)
        ids.add(message_id)
        messages.append(Message(message_id, chat_id, sender_id, sender_name, ts, record["kind"], body, quoted_id))
    return tuple(messages)


def _load_seed(path: Path, participants: set[str]) -> SeedData:
    if not path.exists():
        return SeedData({}, (), ())
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Could not load {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise InputError(f"{path}: seed must be a JSON object")
    circles = raw.get("circles", {})
    organisations = raw.get("organisations", [])
    goals = raw.get("goals", [])
    valid_circles = {"spouse", "child", "parent", "in-law", "sibling", "relative", "friend", "colleague", "client", "vendor", "partner", "staff"}
    if not isinstance(circles, dict) or not all(isinstance(k, str) and v in valid_circles for k, v in circles.items()):
        raise InputError(f"{path}: circles must map IDs to valid circle values")
    unknown = set(circles) - participants
    if unknown:
        raise InputError(f"{path}: seeded circles refer to unknown participants: {sorted(unknown)}")
    if not isinstance(organisations, list) or not all(isinstance(value, str) and value for value in organisations):
        raise InputError(f"{path}: organisations must be a list of names")
    if not isinstance(goals, list) or not all(isinstance(value, dict) for value in goals):
        raise InputError(f"{path}: goals must be a list of objects")
    return SeedData(dict(circles), tuple(organisations), tuple(goals))


def load_input(folder: str | Path) -> InputData:
    folder = Path(folder)
    chats = _load_chats(folder / "chats.jsonl")
    participants = {participant for chat in chats for participant in chat.participants if participant != "ME"}
    messages = _load_messages(folder / "messages.jsonl", {chat.chat_id for chat in chats})
    return InputData(chats, messages, _load_seed(folder / "seed.json", participants))
