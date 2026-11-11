import unittest
import http.client
from unittest.mock import patch

from xpert.indexes import ConversationIndex, ParticipantRegistry
from xpert.ingestion import load_input
from xpert.pipeline import build_baseline
from xpert.extraction import normalise_group_output, normalise_model_output, normalise_people_output
from xpert.llm import LlmError, ask_json


class FoundationTests(unittest.TestCase):
    def test_example_loads_indexes_and_preserves_seed_circle(self):
        data = load_input("example")
        registry = ParticipantRegistry.build(data)
        index = ConversationIndex.build(data)
        output = build_baseline(data, registry, index)
        circles = {person["id"]: person["circle"] for person in output["people"]}
        self.assertEqual(circles["P01"], "spouse")
        self.assertEqual(index.messages_by_chat["C01"][0].id, "M001")
        self.assertEqual(len(output["chats"]), len(data.chats))

    def test_model_output_requires_candidate_source_and_known_ids(self):
        data = load_input("example")
        raw = {"goals": [{"title": "Run", "area": "health", "type": "habit", "horizon": "3mo", "evidence": ["M026"]}], "open_loops": [{"title": "Unsupported", "type": "commitment", "owner": "ME", "chat_id": "C01", "source_msg": "M001", "status": "open"}]}
        goals, loops = normalise_model_output(raw, data, [{"source_msg": "M002"}])
        self.assertEqual(goals[0]["evidence"], ["M026"])
        self.assertEqual(loops, [])

    def test_people_output_cannot_override_seed_or_invent_people(self):
        raw = {"people": [
            {"id": "P01", "circle": "vendor"},
            {"id": "P02", "circle": "client"},
            {"id": "UNKNOWN", "circle": "friend"},
        ]}
        self.assertEqual(
            normalise_people_output(raw, {"P01", "P02"}, {"P01"}),
            {"P02": "client"},
        )

    def test_group_output_requires_group_participant_or_known_organisation(self):
        data = load_input("example")
        chats = {chat.chat_id: chat for chat in data.chats}
        raw = {"groups": [
            {"chat_id": "C03", "target": "P04"},
            {"chat_id": "C03", "target": "Home"},
            {"chat_id": "C03", "target": "P01"},
            {"chat_id": "C03", "target": "ME"},
            {"chat_id": "UNKNOWN", "target": "Home"},
        ]}
        self.assertEqual(normalise_group_output(raw, chats, {"Acme"}), {"C03": "P01"})

    def test_empty_model_content_is_a_safe_llm_error(self):
        response = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self):
                import json
                return json.dumps(response).encode()
        with patch("xpert.llm.configured", return_value=True), patch("xpert.llm.urllib.request.urlopen", return_value=FakeResponse()), patch("xpert.llm.time.sleep"), patch.dict("os.environ", {"LLM_BASE_URL": "https://example.test/v1", "LLM_API_KEY": "key", "LLM_MODEL": "model"}):
            with self.assertRaises(LlmError):
                ask_json("system", "prompt")

    def test_disconnected_model_is_a_safe_llm_error(self):
        with patch("xpert.llm.configured", return_value=True), patch("xpert.llm.urllib.request.urlopen", side_effect=http.client.RemoteDisconnected("closed")), patch("xpert.llm.time.sleep"), patch.dict("os.environ", {"LLM_BASE_URL": "https://example.test/v1", "LLM_API_KEY": "key", "LLM_MODEL": "model", "XPERT_LLM_RETRIES": "1"}):
            with self.assertRaises(LlmError):
                ask_json("system", "prompt")


if __name__ == "__main__":
    unittest.main()
