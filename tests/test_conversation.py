from __future__ import annotations

from dataclasses import dataclass

import pytest

from mlx_dev_agent.conversation import Conversation


@dataclass
class DummyTokenizer:
    bos_token: str | None = "<s>"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


def test_conversation_adds_system_prompt(tokenizer: DummyTokenizer) -> None:
    convo = Conversation(tokenizer=tokenizer, max_context_tokens=10, system_prompt="system")
    assert convo.messages[0] == {"role": "system", "content": "system"}


def test_conversation_truncates_to_respect_limit(tokenizer: DummyTokenizer) -> None:
    convo = Conversation(tokenizer=tokenizer, max_context_tokens=20, system_prompt="sys")
    for idx in range(6):
        convo.add_user(f"user message {idx:02d}")
        convo.add_assistant(f"assistant reply {idx:02d}")

    trimmed = list(convo.trimmed_messages())
    encoded_length = len(tokenizer.encode(convo.prompt(add_generation_prompt=False)))
    assert encoded_length <= 20
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) <= 2
