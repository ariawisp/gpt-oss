import json
import os
import signal
import time
from dataclasses import dataclass
from typing import List, Optional

import pytest
from fastapi.testclient import TestClient
from openai_harmony import HarmonyEncodingName, HarmonyError, StreamState, load_harmony_encoding

from gpt_oss.responses_api.api_server import create_api_server


FINAL_MESSAGE_TEXT = "<|channel|>final<|message|>Hey there<|return|>"
FUNCTION_CALL_ARGUMENTS = '{"location": "Boston"}'
FUNCTION_CALL_TEXT = (
    "<|channel|>commentary<|recipient|>functions.get_weather<|message|>"
    f"{FUNCTION_CALL_ARGUMENTS}<|return|>"
)


def _timeout_handler(signum, frame):
    raise TimeoutError


harmony_available = False
if os.getenv("USE_REAL_HARMONY_ENCODING") == "1":
    try:
        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(1)
        try:
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            harmony_available = True
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    except (HarmonyError, TimeoutError, RuntimeError):
        harmony_available = False


if not harmony_available:

    @dataclass
    class FakeContent:
        text: str


    @dataclass
    class FakeMessage:
        channel: Optional[str]
        recipient: Optional[str]
        content: List[FakeContent]


    @dataclass
    class FakeParsedEntry:
        channel: Optional[str]
        recipient: Optional[str]
        text: str

        def to_dict(self) -> dict:
            payload = {
                "channel": self.channel,
                "content": [{"type": "text", "text": self.text}],
            }
            if self.recipient:
                payload["recipient"] = self.recipient
            return payload


    class FakeEncoding:
        START_FINAL = 1
        START_FUNCTION_CALL = 2
        END = 0

        def __init__(self) -> None:
            self.token_meta = {
                self.START_FINAL: {"channel": "final", "recipient": None},
                self.START_FUNCTION_CALL: {
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                },
            }

        def encode(
            self, text: str, allowed_special: Optional[str] = None
        ) -> List[int]:  # noqa: D401
            final_prefix = "<|channel|>final<|message|>"
            commentary_prefix = "<|channel|>commentary<|recipient|>"
            suffix = "<|return|>"

            if text.startswith(final_prefix) and text.endswith(suffix):
                body = text[len(final_prefix) : -len(suffix)]
                return [self.START_FINAL] + [ord(c) for c in body] + [self.END]

            if text.startswith(commentary_prefix) and text.endswith(suffix):
                remainder = text[len(commentary_prefix) : -len(suffix)]
                recipient, message = remainder.split("<|message|>", 1)
                self.token_meta[self.START_FUNCTION_CALL] = {
                    "channel": "commentary",
                    "recipient": recipient,
                }
                return [self.START_FUNCTION_CALL] + [ord(c) for c in message] + [self.END]

            raise ValueError("Unsupported text for fake encoding")

        def render_conversation_for_completion(self, conversation, role, config=None):
            return []

        def parse_messages_from_completion_tokens(
            self, tokens: List[int], role
        ) -> List[FakeParsedEntry]:
            entries: List[FakeParsedEntry] = []
            channel: Optional[str] = None
            recipient: Optional[str] = None
            buffer = ""
            for token in tokens:
                if token in self.token_meta:
                    if channel is not None:
                        entries.append(FakeParsedEntry(channel, recipient, buffer))
                    meta = self.token_meta[token]
                    channel = meta["channel"]
                    recipient = meta["recipient"]
                    buffer = ""
                elif token == self.END:
                    if channel is not None or buffer:
                        entries.append(FakeParsedEntry(channel, recipient, buffer))
                    channel = None
                    recipient = None
                    buffer = ""
                else:
                    buffer += chr(token)

            if channel is not None or buffer:
                entries.append(FakeParsedEntry(channel, recipient, buffer))

            return entries

        def stop_tokens_for_assistant_actions(self) -> set:
            return {self.END}

        def decode_utf8(self, tokens: List[int]) -> str:
            chars = []
            for token in tokens:
                if token in self.token_meta or token == self.END:
                    continue
                chars.append(chr(token))
            return "".join(chars)


    class FakeStreamableParser:
        def __init__(self, encoding: FakeEncoding, role) -> None:
            self.encoding = encoding
            self.role = role
            self.state = StreamState.EXPECT_START
            self.messages: List[FakeMessage] = []
            self.current_channel: Optional[str] = None
            self.current_recipient: Optional[str] = None
            self.last_content_delta: str = ""
            self._current_channel: Optional[str] = None
            self._current_recipient: Optional[str] = None
            self._current_content: str = ""

        def _finalize_current_message(self) -> None:
            content = [FakeContent(self._current_content)]
            self.messages.append(
                FakeMessage(
                    channel=self._current_channel,
                    recipient=self._current_recipient,
                    content=content,
                )
            )
            self._current_channel = None
            self._current_recipient = None
            self._current_content = ""

        def process(self, token: int) -> None:
            if token in self.encoding.token_meta:
                if self._current_channel is not None:
                    self._finalize_current_message()
                meta = self.encoding.token_meta[token]
                self._current_channel = meta["channel"]
                self._current_recipient = meta["recipient"]
                self.current_channel = self._current_channel
                self.current_recipient = self._current_recipient
                self._current_content = ""
                self.last_content_delta = ""
                self.state = StreamState.HEADER
            elif token == self.encoding.END:
                if self._current_channel is not None:
                    self._finalize_current_message()
                self.state = StreamState.EXPECT_START
                self.current_channel = None
                self.current_recipient = None
                self.last_content_delta = ""
            else:
                char = chr(token)
                self._current_content += char
                self.last_content_delta = char
                self.current_channel = self._current_channel
                self.current_recipient = self._current_recipient
                self.state = StreamState.CONTENT


    encoding = FakeEncoding()

    from gpt_oss.responses_api import api_server as api_server_module

    api_server_module.StreamableParser = FakeStreamableParser


fake_tokens = encoding.encode(FINAL_MESSAGE_TEXT, allowed_special="all")
function_call_tokens = encoding.encode(FUNCTION_CALL_TEXT, allowed_special="all")

token_queue = fake_tokens.copy()


def stub_infer_next_token(
    tokens: list[int], temperature: float = 0.0, new_request: bool = False
) -> int:
    global token_queue
    next_tok = token_queue.pop(0)
    if len(token_queue) == 0:
        token_queue = fake_tokens.copy()
    time.sleep(0.1)
    return next_tok


@pytest.fixture
def test_client():
    return TestClient(
        create_api_server(infer_next_token=stub_infer_next_token, encoding=encoding)
    )


def read_sse_events(response) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current: dict[str, str] = {}
    completed = False
    for raw_line in response.iter_lines():
        if raw_line is None:
            continue
        line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
        if line == "":
            if current:
                event_type = current.get("event", "")
                data_payload = current.get("data")
                data: dict = {}
                if data_payload:
                    data = json.loads(data_payload)
                events.append((event_type, data))
                if event_type == "response.completed":
                    completed = True
                current = {}
            if completed:
                break
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            current["event"] = line.split("event:", 1)[1].strip()
        elif line.startswith("data:"):
            data_line = line.split("data:", 1)[1].strip()
            if "data" in current and current["data"]:
                current["data"] += "\n" + data_line
            else:
                current["data"] = data_line

    if current:
        event_type = current.get("event", "")
        data_payload = current.get("data")
        data: dict = {}
        if data_payload:
            data = json.loads(data_payload)
        events.append((event_type, data))

    return events


def test_health_check(test_client):
    response = test_client.post(
        "/v1/responses",
        json={
            "model": "gpt-oss-120b",
            "input": "Hello, world!",
        },
    )
    print(response.json())
    assert response.status_code == 200


def test_function_call_event_sequence(test_client):
    global fake_tokens, token_queue

    original_fake_tokens = fake_tokens.copy()
    try:
        fake_tokens = function_call_tokens.copy()
        token_queue = function_call_tokens.copy()

        with test_client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "gpt-oss-120b",
                "input": "Hello, world!",
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            events = read_sse_events(response)

    finally:
        fake_tokens = original_fake_tokens
        token_queue = original_fake_tokens.copy()

    event_types = [event_type for event_type, _ in events]

    assert "response.output_item.added" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert "response.output_item.done" in event_types
    assert "response.completed" in event_types

    added_index = event_types.index("response.output_item.added")
    delta_indices = [
        idx
        for idx, event_type in enumerate(event_types)
        if event_type == "response.function_call_arguments.delta"
    ]
    done_index = event_types.index("response.output_item.done")
    completed_index = event_types.index("response.completed")

    assert added_index < delta_indices[0] < done_index < completed_index

    argument_stream = "".join(
        data.get("delta", "")
        for event_type, data in events
        if event_type == "response.function_call_arguments.delta"
    )
    assert argument_stream == FUNCTION_CALL_ARGUMENTS

    added_event = next(
        data for event_type, data in events if event_type == "response.output_item.added"
    )
    assert added_event.get("item", {}).get("type") == "function_call"

    post_done_events = event_types[done_index + 1 : completed_index]
    assert "response.output_text.delta" not in post_done_events
