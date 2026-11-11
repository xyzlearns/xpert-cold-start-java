"""Typed records for input data. The standard library keeps clean-run dependencies at zero."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chat:
    chat_id: str
    name: str
    is_group: bool
    participants: tuple[str, ...]


@dataclass(frozen=True)
class Message:
    id: str
    chat_id: str
    sender_id: str
    sender_name: str
    ts: int | float
    kind: str
    body: str
    quoted_id: str | None


@dataclass(frozen=True)
class SeedData:
    circles: dict[str, str]
    organisations: tuple[str, ...]
    goals: tuple[dict, ...]


@dataclass(frozen=True)
class InputData:
    chats: tuple[Chat, ...]
    messages: tuple[Message, ...]
    seed: SeedData
