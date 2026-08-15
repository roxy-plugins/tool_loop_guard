from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

import plugin as tool_loop_guard
from agent.plugin_composition import CompositionRoot, Context, PluginRuntime
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.tool_hooks.executor import ToolExecutor
from agent.tool_hooks.types import ToolExecutionRequest
from agent.tools.events import TOOL_RESULT, ToolResult
from bus.event_bus import EventBus


def _request(
    *,
    call_id: str,
    command: str,
    session_key: str = "cli:1",
    tool_name: str = "shell",
    tool_batch: tuple[dict[str, Any], ...] = (),
    tool_batch_index: int = 0,
) -> ToolExecutionRequest:
    return ToolExecutionRequest(
        call_id=call_id,
        tool_name=tool_name,
        arguments={"command": command},
        source="passive",
        session_key=session_key,
        channel="cli",
        chat_id="1",
        tool_batch=tool_batch,
        tool_batch_index=tool_batch_index,
    )


async def _mount_guard(
    tmp_path: Path,
    *,
    config: object | None = None,
) -> tuple[CompositionRoot, RuntimeSnapshotStore]:
    root = CompositionRoot("tool-loop-guard-test")
    _ = await root.mount(
        lambda ctx: tool_loop_guard.apply(ctx, config or {}),
        name="tool_loop_guard",
        runtime=PluginRuntime(
            plugin_id="tool_loop_guard",
            plugin_dir=tmp_path / "plugin",
            data_dir=tmp_path / "plugin-data" / "tool_loop_guard",
            workspace=tmp_path / "workspace",
            config=config or {},
        ),
    )
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    return root, store


def test_v3_namespace_is_loadable() -> None:
    loaded = ComposablePlugin.from_module(tool_loop_guard)

    assert loaded.name == "tool_loop_guard"
    assert loaded.version == "2.0.0"
    assert loaded.inject == ()


def test_repeat_limit_preserves_v2_fallbacks() -> None:
    assert tool_loop_guard._repeat_limit({}) == 3
    assert tool_loop_guard._repeat_limit({"repeat_limit": 1}) == 2
    assert tool_loop_guard._repeat_limit({"repeat_limit": "5"}) == 5
    assert tool_loop_guard._repeat_limit({"repeat_limit": object()}) == 3


@pytest.mark.asyncio
async def test_denies_on_third_repeat_without_third_invocation(tmp_path: Path) -> None:
    root, store = await _mount_guard(tmp_path)
    invoked: list[str] = []

    async def invoke(_: str, arguments: dict[str, Any]) -> str:
        invoked.append(str(arguments["command"]))
        return "ok"

    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        results = [
            await ToolExecutor().execute(
                _request(call_id=f"call-{index}", command="echo hi"),
                invoke,
            )
            for index in range(1, 4)
        ]
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()

    assert [item.status for item in results] == ["success", "success", "denied"]
    assert "连续重复调用工具 3 次" in str(results[-1].output)
    assert invoked == ["echo hi", "echo hi"]
    await root.dispose()


@pytest.mark.asyncio
async def test_changed_arguments_and_sessions_have_independent_state(
    tmp_path: Path,
) -> None:
    root, store = await _mount_guard(tmp_path, config={"repeat_limit": 2})

    async def invoke(_: str, __: dict[str, Any]) -> str:
        return "ok"

    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        first = await ToolExecutor().execute(
            _request(call_id="one", command="echo 1"),
            invoke,
        )
        changed = await ToolExecutor().execute(
            _request(call_id="two", command="echo 2"),
            invoke,
        )
        other_session = await ToolExecutor().execute(
            _request(
                call_id="three",
                command="echo 2",
                session_key="cli:2",
            ),
            invoke,
        )
        denied = await ToolExecutor().execute(
            _request(call_id="four", command="echo 2"),
            invoke,
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()

    assert [first.status, changed.status, other_session.status, denied.status] == [
        "success",
        "success",
        "success",
        "denied",
    ]
    await root.dispose()


@pytest.mark.asyncio
async def test_excluded_batch_call_does_not_own_repeat_count(tmp_path: Path) -> None:
    root, store = await _mount_guard(tmp_path, config={"repeat_limit": 2})
    batch = (
        {"name": "task_output", "arguments": {"execution_id": 1}},
        {
            "name": "shell",
            "arguments": {
                "command": "echo hi",
                "nested": {"items": [1, {"name": "same"}]},
            },
        },
    )
    changed_batch = (
        batch[0],
        {
            "name": "shell",
            "arguments": {
                "command": "echo hi",
                "nested": {"items": [1, {"name": "changed"}]},
            },
        },
    )
    invoked: list[int] = []

    async def invoke(_: str, __: dict[str, Any]) -> str:
        invoked.append(1)
        return "ok"

    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        excluded = await ToolExecutor().execute(
            _request(
                call_id="batch-excluded",
                command="",
                tool_name="task_output",
                tool_batch=batch,
                tool_batch_index=0,
            ),
            invoke,
        )
        first = await ToolExecutor().execute(
            _request(
                call_id="batch-first",
                command="echo hi",
                tool_batch=batch,
                tool_batch_index=1,
            ),
            invoke,
        )
        denied = await ToolExecutor().execute(
            _request(
                call_id="batch-denied",
                command="echo hi",
                tool_batch=batch,
                tool_batch_index=1,
            ),
            invoke,
        )
        changed = await ToolExecutor().execute(
            _request(
                call_id="batch-changed",
                command="echo hi",
                tool_batch=changed_batch,
                tool_batch_index=1,
            ),
            invoke,
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()

    assert [excluded.status, first.status, denied.status, changed.status] == [
        "success",
        "success",
        "denied",
        "success",
    ]
    assert len(invoked) == 3
    await root.dispose()


@pytest.mark.asyncio
async def test_preflight_counts_without_publishing_result(tmp_path: Path) -> None:
    root = CompositionRoot("tool-loop-guard-preflight")
    _ = await root.mount(
        lambda ctx: tool_loop_guard.apply(ctx, {"repeat_limit": 2}),
        name="tool_loop_guard",
        runtime=PluginRuntime(
            plugin_id="tool_loop_guard",
            plugin_dir=tmp_path / "plugin",
            data_dir=tmp_path / "plugin-data" / "tool_loop_guard",
            workspace=tmp_path / "workspace",
            config={"repeat_limit": 2},
        ),
    )
    observed: list[ToolResult] = []

    async def observe(ctx: Context) -> None:
        _ = await ctx.on(TOOL_RESULT, observed.append)

    _ = await root.mount(observe, name="observer")
    store = RuntimeSnapshotStore()
    store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        first = await ToolExecutor().preflight(
            _request(call_id="preflight-one", command="echo hi")
        )
        denied = await ToolExecutor().preflight(
            _request(call_id="preflight-two", command="echo hi")
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()

    assert [first.status, denied.status] == ["success", "denied"]
    assert observed == []
    await root.dispose()


@pytest.mark.asyncio
async def test_manager_snapshot_owns_loop_state_and_cleanup(tmp_path: Path) -> None:
    plugin_home = tmp_path / "plugins"
    plugin_home.mkdir()
    _ = shutil.copytree(
        Path(__file__).parents[1],
        plugin_home / "tool_loop_guard",
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
        ),
    )
    manager = PluginManager(
        plugin_dirs=[plugin_home],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "plugin-home" / "cache",
    )
    await manager.load_all()
    generation = manager.generation("tool_loop_guard")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert isinstance(generation.instance, ComposablePlugin)
    root = snapshot.composition_root
    assert root is not None
    assert snapshot.composition_topology is not None
    assert snapshot.composition_topology.listeners == (
        "serial:tool.execution.authorize"
        "[bail=akashic.tool-deny-reason.v1]:tool_loop_guard",
    )

    async def invoke(_: str, __: dict[str, Any]) -> str:
        return "ok"

    lease = manager._snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        results = [
            await ToolExecutor().execute(
                _request(call_id=f"manager-{index}", command="echo hi"),
                invoke,
            )
            for index in range(3)
        ]
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    assert [item.status for item in results] == ["success", "success", "denied"]
    await manager.terminate_all()
    assert root.topology_view().listeners == ()
    assert root.receipt().effects == ()
