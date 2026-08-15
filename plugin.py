from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from agent.plugin_composition import Bail, Context
from agent.tools.events import TOOL_EXECUTION_AUTHORIZE, ToolInput

_DEFAULT_REPEAT_LIMIT = 3
_DENY_PREFIX = "tool_loop_guard:"
_EXCLUDED_TOOLS = frozenset({"task_output", "task_stop"})

api_version = 3
name = "tool_loop_guard"
version = "2.0.0"
desc = "检测连续重复的工具调用并提前截断"
author = "Akashic"
inject: tuple[()] = ()


@dataclass
class _LoopState:
    signature: str = ""
    repeat_count: int = 0


class ToolLoopGuard:
    """Own per-session repeat state and return typed authorization decisions."""

    def __init__(self, repeat_limit: int) -> None:
        self._states: dict[str, _LoopState] = {}
        self._repeat_limit = repeat_limit

    def authorize(self, tool_input: ToolInput) -> Bail[str] | None:
        signature, active_index = self._event_signature(tool_input)
        if not signature or tool_input.tool_batch_index != active_index:
            return None
        state_key = self._state_key(tool_input)
        state = self._states.setdefault(state_key, _LoopState())
        if signature == state.signature:
            state.repeat_count += 1
        else:
            state.signature = signature
            state.repeat_count = 1
        if state.repeat_count < self._repeat_limit:
            return None
        return Bail(
            f"{_DENY_PREFIX}连续重复调用工具 "
            f"{state.repeat_count} 次，已截断并进入收尾。"
        )

    @staticmethod
    def _state_key(tool_input: ToolInput) -> str:
        if tool_input.session_key:
            return f"{tool_input.source}:{tool_input.session_key}"
        return f"{tool_input.source}:{tool_input.channel}:{tool_input.chat_id}"

    @staticmethod
    def _signature(tool_name: str, arguments: Mapping[str, object]) -> str:
        encoded = json.dumps(
            _thaw(arguments),
            ensure_ascii=False,
            sort_keys=True,
        )
        return f"{tool_name}:{encoded}"

    def _event_signature(self, tool_input: ToolInput) -> tuple[str, int]:
        if not tool_input.tool_batch:
            if tool_input.tool_name in _EXCLUDED_TOOLS:
                return "", 0
            return self._signature(tool_input.tool_name, tool_input.arguments), 0

        parts: list[str] = []
        active_index = -1
        for index, tool_call in enumerate(tool_input.tool_batch):
            tool_name = str(tool_call.get("name", ""))
            if tool_name in _EXCLUDED_TOOLS:
                continue
            raw_arguments = tool_call.get("arguments")
            arguments: Mapping[str, object] = {}
            if isinstance(raw_arguments, Mapping):
                arguments = cast(Mapping[str, object], raw_arguments)
            if active_index < 0:
                active_index = index
            parts.append(self._signature(tool_name, arguments))
        if active_index < 0:
            return "", 0
        return "|".join(parts), active_index


async def apply(ctx: Context, config: object) -> None:
    """Register one generation-scoped loop authorizer."""

    guard = ToolLoopGuard(_repeat_limit(config))
    _ = await ctx.on(TOOL_EXECUTION_AUTHORIZE, guard.authorize)


def _repeat_limit(config: object) -> int:
    raw_limit: object = _DEFAULT_REPEAT_LIMIT
    if isinstance(config, Mapping):
        raw_limit = cast(Mapping[object, object], config).get(
            "repeat_limit",
            _DEFAULT_REPEAT_LIMIT,
        )
    try:
        return max(2, int(cast(int | str, raw_limit)))
    except (TypeError, ValueError):
        return _DEFAULT_REPEAT_LIMIT


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): _thaw(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in cast(tuple[object, ...], value)]
    return value
