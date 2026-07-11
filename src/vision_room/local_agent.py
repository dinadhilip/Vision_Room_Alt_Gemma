from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_video_library",
            "description": "Search the user's local indexed video library semantically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cast_into_frame",
            "description": "Cast a character, person, or product into the confirmed local frame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_frame_path": {"type": "string"},
                    "casting_prompt": {"type": "string"},
                    "reference_image_path": {"type": ["string", "null"], "default": None},
                },
                "required": ["casting_prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_video",
            "description": "Compose approved anchor frames into a short conversational video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor_frame_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "narrative_hint": {"type": "string"},
                    "duration_hint_s": {"type": "integer", "minimum": 3, "maximum": 60, "default": 15},
                    "edit_instruction": {"type": ["string", "null"], "default": None},
                    "prior_video_id": {"type": ["string", "null"], "default": None},
                },
                "required": ["narrative_hint"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_storyboard",
            "description": "Generate a comic-style storyboard from a user's story based on matched frames.",
            "parameters": {
                "type": "object",
                "properties": {
                    "story": {"type": "string"},
                    "style": {"type": "string", "default": "comic"},
                },
                "required": ["story"],
            },
        },
    },
]


@dataclass(frozen=True)
class PlannedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class PlannedReply:
    reply: str


class LocalGemmaPlanner:
    """OpenAI-compatible local model planner for Gemma served by litert-lm."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s

    def plan(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, str]],
        state: dict[str, Any],
    ) -> PlannedToolCall | PlannedReply | None:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "system",
                "content": "Current session state, for tool preconditions only: "
                + json.dumps(state, ensure_ascii=True),
            },
            *history,
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]

        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            function = tool_calls[0].get("function", {})
            return PlannedToolCall(
                name=function.get("name", ""),
                arguments=_parse_arguments(function.get("arguments")),
            )

        content = (message.get("content") or "").strip()
        if not content:
            return None
        json_call = _parse_json_tool_call(content)
        if json_call:
            return json_call
        return PlannedReply(reply=content)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_tool_call(content: str) -> PlannedToolCall | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("tool") or parsed.get("name")
    arguments = parsed.get("arguments") or parsed.get("args") or {}
    if isinstance(name, str) and isinstance(arguments, dict):
        return PlannedToolCall(name=name, arguments=arguments)
    return None

