from __future__ import annotations

import pytest

from agent.lifecycle.types import PreToolCtx
from plugin import ToolLoopGuard


@pytest.mark.asyncio
async def test_tool_loop_guard_denies_on_third_repeat() -> None:
    plugin = ToolLoopGuard()
    first = await plugin.detect_repeated_tool_call(
        PreToolCtx(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            tool_name="shell",
            arguments={"command": "echo hi"},
            source="passive",
        )
    )
    second = await plugin.detect_repeated_tool_call(
        PreToolCtx(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            tool_name="shell",
            arguments={"command": "echo hi"},
            source="passive",
        )
    )
    third = await plugin.detect_repeated_tool_call(
        PreToolCtx(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            tool_name="shell",
            arguments={"command": "echo hi"},
            source="passive",
        )
    )
    assert first is None
    assert second is None
    assert third is not None
    assert third.decision == "deny"


@pytest.mark.asyncio
async def test_tool_loop_guard_ignores_changed_arguments() -> None:
    plugin = ToolLoopGuard()
    first = await plugin.detect_repeated_tool_call(
        PreToolCtx(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            tool_name="shell",
            arguments={"command": "echo 1"},
            source="passive",
        )
    )
    second = await plugin.detect_repeated_tool_call(
        PreToolCtx(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            tool_name="shell",
            arguments={"command": "echo 2"},
            source="passive",
        )
    )
    assert first is None
    assert second is None
