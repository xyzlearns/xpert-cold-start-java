"""Low-cost candidate generation and schema-safe extraction normalisation."""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from .indexes import ConversationIndex, ParticipantRegistry
from .models import InputData, Message

COMMITMENT = re.compile(r"\b(i(?:'ll| will| can)|we(?:'ll| will)|will|need to|remind me|please (?:send|check|confirm|share)|can you|could you|when (?:will|can)|waiting for|pending|due|follow up|follow-up)\b", re.I)
EXPLICIT_PROMISE = re.compile(r"\b(i(?:'ll| will)|we(?:'ll| will))\b.*\b(send|share|check|book|call|pay|confirm|update|follow|prepare|revert|approve|deliver|arrange)\b", re.I)
DIRECTED_REQUEST = re.compile(r"\b(can|could|will|would) you\b|\bplease\s+(send|share|check|confirm|approve|call|update)\b", re.I)
PENDING = re.compile(r"\b(waiting|pending|remind|follow[ -]?up|due|any update|when can|when will)\b", re.I)
INCIDENT = re.compile(r"\b(expired|error|unable|not able|blocked|failed|missing|issue|problem|critical)\b", re.I)
QUESTION = re.compile(r"\?|\b(what|which|when|where|how|should|shall|can|could|would|do|are|is)\s+(?:i|we|you|this|that|the)\b", re.I)
INFORMATIONAL_UPDATE = re.compile(r"\b(is now working|already (?:initiated|done|completed|finalized)|we'?ll do (?:it|that|monday)|will be joining|will reach by|will come in|will be there)\b", re.I)
BROADCAST = re.compile(r"https?://|\b(top headlines|read\s*[-:]|ipo|newsletter|subscribe)\b", re.I)
GOAL = re.compile(r"\b(goal|target|plan to|training|learn|lose|save|run \d|grow|launch|by march|this year)\b", re.I)
WORK = re.compile(r"\b(invoice|client|project|proposal|team|meeting|vendor|payment|contract|company|office|launch|deck)\b", re.I)
FAMILY = re.compile(r"\b(mom|mother|dad|father|mama|papa|brother|sister|wife|husband|son|daughter|kid|kids|school)\b", re.I)


def text(message: Message, limit: int = 260) -> str:
    return " ".join(message.body.split())[:limit]


def conversation_summaries(data: InputData, index: ConversationIndex, limit: int = 5) -> list[dict]:
    summaries = []
    for chat in data.chats:
        messages = index.messages_by_chat.get(chat.chat_id, ())
        samples = list(messages[-limit:])
        summaries.append({"chat_id": chat.chat_id, "name": chat.name, "group": chat.is_group, "participants": list(chat.participants), "messages": [{"id": m.id, "sender": m.sender_id, "text": text(m)} for m in samples]})
    return summaries


def heuristic_people(data: InputData, registry: ParticipantRegistry, index: ConversationIndex) -> dict[str, str]:
    circles = dict(data.seed.circles)
    direct_chat = {next(person for person in chat.participants if person != "ME"): chat for chat in data.chats if not chat.is_group and len(chat.participants) == 2}
    for person in registry.participant_ids:
        if person in circles:
            continue
        chat = direct_chat.get(person)
        messages = index.messages_by_sender.get(person, ())
        corpus = " ".join([chat.name if chat else "", *[m.body for m in messages[-8:]]])
        name = " ".join(registry.display_names.get(person, ())).lower()
        if any(word in name for word in ("doctor", "plumber", "driver", "maid", "cook", "teacher")):
            circles[person] = "staff"
        elif WORK.search(corpus):
            circles[person] = "colleague"
        elif FAMILY.search(corpus):
            circles[person] = "relative"
        else:
            circles[person] = "friend"
    return circles


def person_profiles(data: InputData, registry: ParticipantRegistry, index: ConversationIndex) -> list[dict]:
    """Return compact, grounded evidence for relationship classification.

    Names alone are weak evidence in an anonymised export.  Direct-chat titles, the
    groups shared with ME, and a few recent utterances make the distinction between a
    client, vendor, colleague and friend much more reliable while keeping model calls
    bounded.
    """
    direct_chats = {
        next(person for person in chat.participants if person != "ME"): chat
        for chat in data.chats
        if not chat.is_group and len(chat.participants) == 2
    }
    groups_by_person: dict[str, list[str]] = defaultdict(list)
    for chat in data.chats:
        if chat.is_group:
            for person in chat.participants:
                if person != "ME":
                    groups_by_person[person].append(chat.name)
    profiles = []
    for person in registry.participant_ids:
        messages = index.messages_by_sender.get(person, ())
        direct = direct_chats.get(person)
        direct_messages = index.messages_by_chat.get(direct.chat_id, ())[-6:] if direct else ()
        profiles.append({
            "id": person,
            "names": list(registry.display_names.get(person, ()))[:2],
            "direct_chat": direct.name if direct else None,
            "groups": groups_by_person.get(person, [])[:8],
            "messages": [text(message, 110) for message in messages[-4:] if text(message, 110)],
            "direct_dialogue": [
                {"sender": message.sender_id, "text": text(message, 110)}
                for message in direct_messages
                if text(message, 110)
            ],
        })
    return profiles


def normalise_people_output(raw: object, known_people: set[str], seeded: set[str]) -> dict[str, str]:
    """Keep only enum-valid model classifications for unseeded people."""
    if not isinstance(raw, dict):
        return {}
    valid = {"spouse", "child", "parent", "in-law", "sibling", "relative", "friend", "colleague", "client", "vendor", "partner", "staff"}
    out: dict[str, str] = {}
    for person in raw.get("people", []):
        if not isinstance(person, dict):
            continue
        person_id, circle = person.get("id"), person.get("circle")
        if isinstance(person_id, str) and person_id in known_people and person_id not in seeded and circle in valid:
            out[person_id] = circle
    return out


def group_profiles(data: InputData, index: ConversationIndex) -> list[dict]:
    """Compact, evidence-only descriptions of group chats for ownership decisions."""
    profiles = []
    for chat in data.chats:
        if not chat.is_group:
            continue
        messages = index.messages_by_chat.get(chat.chat_id, ())
        profiles.append({
            "chat_id": chat.chat_id,
            "name": chat.name,
            "participants": list(chat.participants),
            "recent_messages": [
                {"sender": message.sender_id, "text": text(message, 140)}
                for message in messages[-6:]
                if text(message, 140)
            ],
        })
    return profiles


def normalise_group_output(raw: object, chats: dict[str, object], known_orgs: set[str]) -> dict[str, str]:
    """Accept only a supplied group and either a known org, Home, or its participant."""
    if not isinstance(raw, dict):
        return {}
    ties: dict[str, str] = {}
    allowed_orgs = known_orgs | {"Home"}
    for item in raw.get("groups", []):
        if not isinstance(item, dict):
            continue
        chat_id, target = item.get("chat_id"), item.get("target")
        chat = chats.get(chat_id) if isinstance(chat_id, str) else None
        if chat is None or not isinstance(target, str):
            continue
        if target in allowed_orgs or (target != "ME" and target in getattr(chat, "participants", ())):
            ties[chat_id] = target
    return ties


def heuristic_organisations(data: InputData, index: ConversationIndex) -> tuple[list[dict], dict[str, str]]:
    """Tie group chats to conservative umbrella organisations.

    WhatsApp group names often contain a client/project name rather than the company
    that owns the relationship.  The seed names and well-established PMA/D2C aliases
    are more reliable than emitting an unrelated organisation for every group.
    """
    organisations: list[dict] = []
    chat_org: dict[str, str] = {}
    by_name: dict[str, str] = {}

    def canonical_label(chat_name: str, corpus: str) -> tuple[str, str, str, str]:
        lower_name = chat_name.lower()
        seed_match = next((name for name in data.seed.organisations if name.lower() in corpus), None)
        if seed_match:
            return seed_match, "company", "work", "member"
        if re.search(r"\b(operations|acq\.?\s*rate)\b", lower_name) and data.seed.organisations:
            return data.seed.organisations[0], "company", "work", "member"
        # These regional/topic variants are chapters of the same D2C community.
        if re.search(r"\bd2c\s+circle\b", lower_name):
            return "D2C Circle", "other", "work", "member"
        if re.search(r"\bd2b\s+connect\b", lower_name):
            return "D2B Connect", "other", "work", "member"
        # Family and household groups are one home context, not independent companies.
        if re.search(r"family|siblings?|sisters?|bro(?:ther|&)|home is where", lower_name):
            return "Home", "residence", "home", "owner"
        return chat_name, "company" if WORK.search(corpus) else "other", "work" if WORK.search(corpus) else "home", "member"

    for chat in data.chats:
        if not chat.is_group:
            continue
        corpus = " ".join([chat.name, *[m.body for m in index.messages_by_chat.get(chat.chat_id, ())[-12:]]]).lower()
        label, org_type, side, role = canonical_label(chat.name, corpus)
        key = re.sub(r"\W+", "", label.lower())
        org_id = by_name.get(key)
        if org_id is None:
            org_id = "org_" + re.sub(r"[^a-z0-9]+", "_", key).strip("_")[:40]
            if not org_id or org_id == "org_":
                org_id = f"org_{chat.chat_id.lower()}"
            by_name[key] = org_id
            organisations.append({"id": org_id, "name": label, "type": org_type, "side": side, "role": role})
        chat_org[chat.chat_id] = org_id
    return organisations, chat_org


def goal_candidates(data: InputData) -> list[dict]:
    candidates = []
    for message in data.messages:
        if GOAL.search(message.body):
            candidates.append({"id": message.id, "chat_id": message.chat_id, "sender": message.sender_id, "text": text(message)})
    return candidates[-80:]


def loop_candidates(data: InputData, index: ConversationIndex) -> list[dict]:
    """Candidate sources only; this intentionally does not claim they are open yet."""
    newest = max((message.ts for message in data.messages), default=0)
    cutoff = newest - 28 * 24 * 3600
    candidates: list[dict] = []
    for chat_id, messages in index.messages_by_chat.items():
        direct_chat = len(index.chats_by_id[chat_id].participants) == 2
        for position, message in enumerate(messages):
            # One-word "?" replies are conversational turns, not standalone assistant tasks.
            direct_question = direct_chat and len(message.body.strip()) >= 18 and bool(QUESTION.search(message.body))
            if message.ts < cutoff or not (COMMITMENT.search(message.body) or INCIDENT.search(message.body) or direct_question):
                continue
            if INFORMATIONAL_UPDATE.search(message.body):
                continue
            # Include enough later messages for the model to see a confirmation or resolution.
            context = messages[max(0, position - 2): position + 7]
            quality = 0
            if EXPLICIT_PROMISE.search(message.body):
                quality += 8
            if DIRECTED_REQUEST.search(message.body):
                quality += 6
            if PENDING.search(message.body):
                quality += 4
            if INCIDENT.search(message.body):
                quality += 5
            if message.sender_id == "ME":
                quality += 1
            # A direct conversation is much more likely to represent something the personal
            # assistant can safely follow up on than an unaddressed ticket in a large group.
            if direct_chat:
                quality += 6
            if direct_question:
                quality += 4
            if BROADCAST.search(message.body):
                quality -= 12
            previous = messages[max(0, position - 2):position]
            # "We'll do it" immediately after a request is usually a confirmation, not the
            # strongest source. Preserve the preceding request for model adjudication instead.
            if EXPLICIT_PROMISE.search(message.body) and any(DIRECTED_REQUEST.search(item.body) for item in previous):
                quality -= 6
            candidates.append({"source_msg": message.id, "chat_id": chat_id, "owner_hint": message.sender_id, "ts": message.ts, "quality": quality, "context": [{"id": item.id, "sender": item.sender_id, "text": text(item)} for item in context]})
    # Message-file order and chat-ID order are not chronological; rank using timestamp and signal.
    return sorted(candidates, key=lambda item: (item["quality"], item["ts"]), reverse=True)[:120]


def fallback_open_loops(candidates: list[dict]) -> list[dict]:
    """Return only high-confidence promises when no model endpoint is usable.

    This is deliberately narrow: an ungrounded false positive costs more than a missed task.
    """
    promise = re.compile(r"\b(i(?:'ll| will)|we(?:'ll| will))\b.*\b(send|share|check|book|call|pay|confirm|update|follow|prepare|make)\b", re.I)
    completed = re.compile(r"\b(sent|shared|booked|called|paid|confirmed|updated|done|completed|finished)\b", re.I)
    loops = []
    for candidate in candidates:
        context = candidate["context"]
        source = next((item for item in context if item["id"] == candidate["source_msg"]), None)
        if not source or not promise.search(source["text"]):
            continue
        later = context[context.index(source) + 1:]
        closing = next((item["id"] for item in later if completed.search(item["text"])), None)
        loops.append({
            "title": source["text"][:180], "type": "commitment", "owner": source["sender"],
            "chat_id": candidate["chat_id"], "source_msg": source["id"], "due": None,
            "status": "done" if closing else "open", "closed_by": closing, "help": None,
        })
    return loops[:25]


def normalise_model_output(raw: object, data: InputData, candidates: list[dict], allowed_goal_ids: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Accept only model claims grounded in supplied candidate message IDs and schema enums."""
    if not isinstance(raw, dict):
        return [], []
    message_ids = {message.id for message in data.messages}
    chat_ids = {chat.chat_id for chat in data.chats}
    people = {person for chat in data.chats for person in chat.participants} - {"ME"}
    candidate_ids = {candidate["source_msg"] for candidate in candidates}
    valid_goals, valid_loops = [], []
    for goal in raw.get("goals", []):
        if not isinstance(goal, dict):
            continue
        evidence = [item for item in goal.get("evidence", []) if item in message_ids and (allowed_goal_ids is None or item in allowed_goal_ids)]
        if goal.get("area") in {"self", "spouse", "kids", "parents", "work", "friends", "experiences", "finances", "health", "home"} and goal.get("type") in {"habit", "milestone", "experience"} and goal.get("horizon") in {"3mo", "1yr", "3yr", "5yr"} and isinstance(goal.get("title"), str) and evidence:
            valid_goals.append({"title": goal["title"][:180], "area": goal["area"], "type": goal["type"], "horizon": goal["horizon"], "evidence": evidence[:4]})
    for loop in raw.get("open_loops", []):
        if not isinstance(loop, dict) or loop.get("source_msg") not in candidate_ids:
            continue
        if loop.get("chat_id") not in chat_ids or loop.get("owner") not in people | {"ME"} or loop.get("type") not in {"commitment", "followup", "deadline", "question"} or loop.get("status") not in {"open", "done", "dropped"} or not isinstance(loop.get("title"), str):
            continue
        item = {"title": loop["title"][:180], "type": loop["type"], "owner": loop["owner"], "chat_id": loop["chat_id"], "source_msg": loop["source_msg"], "due": None, "status": loop["status"], "closed_by": loop.get("closed_by") if loop.get("closed_by") in message_ids else None, "help": None}
        if item["status"] == "open" and isinstance(loop.get("help"), dict):
            action = loop["help"].get("action")
            if action == "assist" and isinstance(loop["help"].get("next_step"), str): item["help"] = {"action": "assist", "next_step": loop["help"]["next_step"][:180], "needs_yes": bool(loop["help"].get("needs_yes", True))}
            elif action == "remind": item["help"] = {"action": "remind", "remind_at": datetime.now(UTC).isoformat()}
            elif action == "leave" and isinstance(loop["help"].get("reason"), str): item["help"] = {"action": "leave", "reason": loop["help"]["reason"][:180]}
        valid_loops.append(item)
    return valid_goals[:10], valid_loops
