"""Conversation management for the MLX agent."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, str]


@dataclass(slots=True)
class Conversation:
    """Keeps track of the dialogue and enforces context limits."""

    tokenizer: Any
    max_context_tokens: int
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._truncate_if_needed()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._truncate_if_needed()

    def prompt(self, add_generation_prompt: bool = True) -> str:
        """Return the chat template string for the current history."""

        return self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    def trimmed_messages(self) -> Iterable[Message]:
        """Return a shallow copy of the current messages list."""

        return list(self.messages)

    def reset(self) -> None:
        base: list[Message] = []
        if self.system_prompt:
            base.append({"role": "system", "content": self.system_prompt})
        self.messages = base

    def _truncate_if_needed(self) -> None:
        if not self.max_context_tokens:
            return
        encoded = self.tokenizer.encode(
            self.prompt(add_generation_prompt=False), add_special_tokens=False
        )
        while len(encoded) > self.max_context_tokens and len(self.messages) > 1:
            drop_index = 1 if self.messages[0]["role"] == "system" else 0
            dropped = self.messages.pop(drop_index)
            if (
                dropped["role"] == "user"
                and len(self.messages) > drop_index
                and self.messages[drop_index]["role"] == "assistant"
            ):
                self.messages.pop(drop_index)
            encoded = self.tokenizer.encode(
                self.prompt(add_generation_prompt=False), add_special_tokens=False
            )
