"""Deterministic conversation and participant indexes used by later extractors."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .models import Chat, InputData, Message


@dataclass(frozen=True)
class ConversationIndex:
    chats_by_id: dict[str, Chat]
    messages_by_id: dict[str, Message]
    messages_by_chat: dict[str, tuple[Message, ...]]
    messages_by_sender: dict[str, tuple[Message, ...]]

    @classmethod
    def build(cls, data: InputData) -> "ConversationIndex":
        by_chat: dict[str, list[tuple[int, Message]]] = defaultdict(list)
        by_sender: dict[str, list[tuple[int, Message]]] = defaultdict(list)
        for position, message in enumerate(data.messages):
            by_chat[message.chat_id].append((position, message))
            by_sender[message.sender_id].append((position, message))
        sort = lambda values: tuple(message for _, message in sorted(values, key=lambda item: (item[1].ts, item[0])))
        return cls(
            {chat.chat_id: chat for chat in data.chats},
            {message.id: message for message in data.messages},
            {chat_id: sort(messages) for chat_id, messages in by_chat.items()},
            {sender_id: sort(messages) for sender_id, messages in by_sender.items()},
        )


@dataclass(frozen=True)
class ParticipantRegistry:
    participant_ids: tuple[str, ...]
    display_names: dict[str, tuple[str, ...]]
    chat_ids: dict[str, tuple[str, ...]]

    @classmethod
    def build(cls, data: InputData) -> "ParticipantRegistry":
        names: dict[str, set[str]] = defaultdict(set)
        chats: dict[str, set[str]] = defaultdict(set)
        for chat in data.chats:
            for participant in chat.participants:
                if participant != "ME":
                    chats[participant].add(chat.chat_id)
        for message in data.messages:
            if message.sender_id != "ME":
                names[message.sender_id].add(message.sender_name)
        ids = sorted(set(chats) | set(names))
        return cls(tuple(ids), {key: tuple(sorted(value)) for key, value in names.items()}, {key: tuple(sorted(value)) for key, value in chats.items()})
