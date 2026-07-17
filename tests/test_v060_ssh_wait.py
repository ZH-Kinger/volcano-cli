"""v0.6.0 regression: `volcano ssh` waits for the target Pod to reach Running
before connecting (and the `--no-wait` escape hatch).

Two surfaces under test, both fully mocked (no real cluster, no real sleep):

* ``cli._wait_pod_running`` — the polling helper: returns as soon as the target
  Pod is Running; on ``sdk.status`` error it returns immediately (放行, never
  blocks the connection); it stops after ``timeout`` without raising; and a
  ``KeyboardInterrupt`` from ``time.sleep`` aborts the wait cleanly.
* ``cli.ssh_cmd`` — calls ``_wait_pod_running`` once by default and *not at all*
  under ``--no-wait``.

``cli.time.sleep`` is monkeypatched to a counter so nothing actually sleeps, and
``cli.sdk.status`` is replaced by a programmable sequence of phase snapshots.
Only tests/ is touched — no source edits.
"""

import shutil
import subprocess

import pytest
from typer.testing import CliRunner

import volcano.cli as cli

runner = CliRunner()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sleep_counter(box):
    """Return a fake time.sleep that counts calls (and never actually sleeps)."""
    def _sleep(_seconds):
        box["sleeps"] = box.get("sleeps", 0) + 1
    return _sleep


def _status_sequence(snapshots, calls_box=None):
    """Return a fake sdk.status that yields snapshots in order, repeating the last.

    Each snapshot is a full ``status`` dict, e.g. ``{"pods": [{"name":..,"phase":..}]}``.
    """
    state = {"i": 0}

    def _status(name, team=None):
        if calls_box is not None:
            calls_box["calls"] = calls_box.get("calls", 0) + 1
        idx = min(state["i"], len(snapshots) - 1)
        state["i"] += 1
        return snapshots[idx]

    return _status


def _pods(*items):
    """items: (name, phase) tuples -> {"pods": [...]}."""
    return {"pods": [{"name": n, "phase": ph} for n, ph in items]}


@pytest.fixture
def patch_sleep(monkeypatch):
    """Replace cli.time.sleep with a counter; hand back the count box."""
    box = {}
    monkeypatch.setattr(cli.time, "sleep", _sleep_counter(box))
    return box


# --------------------------------------------------------------------------- #
# _wait_pod_running: case ① already Running -> immediate return, no sleep
# --------------------------------------------------------------------------- #
def test_wait_returns_immediately_when_running(patch_sleep, monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda name, team=None: _pods(("j-worker-0", "Running")),
    )
    cli._wait_pod_running("j", team="teamx")
    assert patch_sleep.get("sleeps", 0) == 0


# --------------------------------------------------------------------------- #
# _wait_pod_running: case ② Pending a few rounds then Running
# --------------------------------------------------------------------------- #
def test_wait_pending_then_running(patch_sleep, monkeypatch):
    calls = {}
    seq = _status_sequence(
        [
            _pods(("j-worker-0", "Pending")),
            _pods(("j-worker-0", "Pending")),
            _pods(("j-worker-0", "Running")),
        ],
        calls_box=calls,
    )
    monkeypatch.setattr(cli.sdk, "status", seq)
    cli._wait_pod_running("j", team="teamx")
    # 2 pending rounds slept, then the 3rd poll saw Running and returned.
    assert patch_sleep.get("sleeps", 0) == 2
    assert calls["calls"] == 3


# --------------------------------------------------------------------------- #
# _wait_pod_running: case ③ Pending forever -> stops at timeout, no exception
# --------------------------------------------------------------------------- #
def test_wait_times_out_without_raising(patch_sleep, monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda name, team=None: _pods(("j-worker-0", "Pending")),
    )
    # timeout=12, sleep step is 3 -> waited walks 0,3,6,9 then hits 12 and stops.
    cli._wait_pod_running("j", team="teamx", timeout=12)
    assert patch_sleep.get("sleeps", 0) == 12 // 3  # == 4


def test_wait_default_timeout_is_120(patch_sleep, monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda name, team=None: _pods(("j-worker-0", "Pending")),
    )
    cli._wait_pod_running("j", team="teamx")  # default timeout=120
    assert patch_sleep.get("sleeps", 0) == 120 // 3  # == 40


# --------------------------------------------------------------------------- #
# _wait_pod_running: case ④ sdk.status raises ->放行 (immediate return, no sleep)
# --------------------------------------------------------------------------- #
def test_wait_returns_on_status_exception(patch_sleep, monkeypatch):
    def boom(name, team=None):
        raise RuntimeError("apiserver unreachable")

    monkeypatch.setattr(cli.sdk, "status", boom)
    # Must not raise, must not block.
    cli._wait_pod_running("j", team="teamx")
    assert patch_sleep.get("sleeps", 0) == 0


def test_wait_returns_on_status_exception_second_round(patch_sleep, monkeypatch):
    """A transient status error mid-wait also releases (放行) rather than looping."""
    state = {"i": 0}

    def flaky(name, team=None):
        state["i"] += 1
        if state["i"] == 1:
            return _pods(("j-worker-0", "Pending"))
        raise RuntimeError("blip")

    monkeypatch.setattr(cli.sdk, "status", flaky)
    cli._wait_pod_running("j", team="teamx")
    # one Pending round (1 sleep), then the exception releases.
    assert patch_sleep.get("sleeps", 0) == 1


# --------------------------------------------------------------------------- #
# _wait_pod_running: case ⑤ worker=N only counts <name>-worker-N
# --------------------------------------------------------------------------- #
def test_wait_worker_ignores_other_running_pods(patch_sleep, monkeypatch):
    """worker=1: worker-0 Running must NOT count; only worker-1 Running is ready."""
    seq = _status_sequence([
        # worker-0 already Running, but target worker-1 is still Pending.
        _pods(("j-worker-0", "Running"), ("j-worker-1", "Pending")),
        _pods(("j-worker-0", "Running"), ("j-worker-1", "Pending")),
        _pods(("j-worker-0", "Running"), ("j-worker-1", "Running")),
    ])
    monkeypatch.setattr(cli.sdk, "status", seq)
    cli._wait_pod_running("j", team="teamx", worker=1)
    # It kept waiting through 2 Pending rounds despite worker-0 being Running.
    assert patch_sleep.get("sleeps", 0) == 2


def test_wait_worker_never_ready_times_out(patch_sleep, monkeypatch):
    """worker-1 stays Pending while worker-0 is Running -> waits full timeout."""
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda name, team=None: _pods(("j-worker-0", "Running"), ("j-worker-1", "Pending")),
    )
    cli._wait_pod_running("j", team="teamx", worker=1, timeout=6)
    assert patch_sleep.get("sleeps", 0) == 6 // 3  # == 2


def test_wait_worker_target_absent_times_out(patch_sleep, monkeypatch):
    """Target worker pod not present at all -> no match, waits full timeout, no raise."""
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda name, team=None: _pods(("j-worker-0", "Running")),
    )
    cli._wait_pod_running("j", team="teamx", worker=3, timeout=6)
    assert patch_sleep.get("sleeps", 0) == 2


def test_wait_worker_zero_matches(patch_sleep, monkeypatch):
    """worker=0 is a real target (not falsy-collapsed to 'no target')."""
    seq = _status_sequence([
        _pods(("j-worker-0", "Pending"), ("j-worker-1", "Running")),
        _pods(("j-worker-0", "Running"), ("j-worker-1", "Running")),
    ])
    monkeypatch.setattr(cli.sdk, "status", seq)
    cli._wait_pod_running("j", team="teamx", worker=0)
    # worker-1 Running must not satisfy worker=0; one Pending round then Running.
    assert patch_sleep.get("sleeps", 0) == 1


# --------------------------------------------------------------------------- #
# _wait_pod_running: case ⑥ explicit pod name
# --------------------------------------------------------------------------- #
def test_wait_explicit_pod_only(patch_sleep, monkeypatch):
    seq = _status_sequence([
        _pods(("other-pod", "Running"), ("my-pod", "Pending")),
        _pods(("other-pod", "Running"), ("my-pod", "Running")),
    ])
    monkeypatch.setattr(cli.sdk, "status", seq)
    cli._wait_pod_running("j", team="teamx", pod="my-pod")
    assert patch_sleep.get("sleeps", 0) == 1


def test_wait_pod_wins_over_worker(patch_sleep, monkeypatch):
    """If both pod and worker given, pod is the target (mirrors _first_running_pod)."""
    seq = _status_sequence([
        # explicit pod Pending; the worker-derived name is Running (must be ignored).
        _pods(("explicit", "Pending"), ("j-worker-2", "Running")),
        _pods(("explicit", "Running"), ("j-worker-2", "Running")),
    ])
    monkeypatch.setattr(cli.sdk, "status", seq)
    cli._wait_pod_running("j", team="teamx", pod="explicit", worker=2)
    assert patch_sleep.get("sleeps", 0) == 1


# --------------------------------------------------------------------------- #
# _wait_pod_running: case ⑦ no target -> any Running satisfies
# --------------------------------------------------------------------------- #
def test_wait_no_target_any_running(patch_sleep, monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda name, team=None: _pods(("a", "Pending"), ("b", "Running")),
    )
    cli._wait_pod_running("j", team="teamx")
    assert patch_sleep.get("sleeps", 0) == 0


def test_wait_no_target_all_pending_then_one_running(patch_sleep, monkeypatch):
    seq = _status_sequence([
        _pods(("a", "Pending"), ("b", "Pending")),
        _pods(("a", "Pending"), ("b", "Running")),
    ])
    monkeypatch.setattr(cli.sdk, "status", seq)
    cli._wait_pod_running("j", team="teamx")
    assert patch_sleep.get("sleeps", 0) == 1


def test_wait_no_pods_times_out(patch_sleep, monkeypatch):
    """Empty pod list (still queued) -> waits full timeout without raising."""
    monkeypatch.setattr(cli.sdk, "status", lambda name, team=None: {"pods": []})
    cli._wait_pod_running("j", team="teamx", timeout=6)
    assert patch_sleep.get("sleeps", 0) == 2


def test_wait_missing_pods_key(patch_sleep, monkeypatch):
    """status dict without a 'pods' key -> .get default [], waits then times out."""
    monkeypatch.setattr(cli.sdk, "status", lambda name, team=None: {})
    cli._wait_pod_running("j", team="teamx", timeout=6)
    assert patch_sleep.get("sleeps", 0) == 2


# --------------------------------------------------------------------------- #
# _wait_pod_running: KeyboardInterrupt during sleep -> clean abort
# --------------------------------------------------------------------------- #
def test_wait_keyboardinterrupt_aborts(monkeypatch, capsys):
    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt)
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda name, team=None: _pods(("j-worker-0", "Pending")),
    )
    # Must not propagate KeyboardInterrupt.
    cli._wait_pod_running("j", team="teamx")
    out = capsys.readouterr().out
    assert "已中断等待" in out


# --------------------------------------------------------------------------- #
# _wait_pod_running: progress announcements
# --------------------------------------------------------------------------- #
def test_wait_announces_progress_and_ready(patch_sleep, monkeypatch, capsys):
    seq = _status_sequence([
        _pods(("j-worker-0", "Pending")),
        _pods(("j-worker-0", "Running")),
    ])
    monkeypatch.setattr(cli.sdk, "status", seq)
    cli._wait_pod_running("j", team="teamx")
    out = capsys.readouterr().out
    assert "容器还没就绪" in out          # announced while Pending
    assert "当前 Pending" in out          # includes the observed phase
    assert "容器已 Running" in out        # announced==True path prints the ready line


def test_wait_immediate_running_no_announcement(patch_sleep, monkeypatch, capsys):
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda name, team=None: _pods(("j-worker-0", "Running")),
    )
    cli._wait_pod_running("j", team="teamx")
    out = capsys.readouterr().out
    # Never announced waiting, so no "ready" follow-up line either.
    assert "容器还没就绪" not in out
    assert "容器已 Running" not in out


# --------------------------------------------------------------------------- #
# ssh_cmd: --no-wait toggles _wait_pod_running
# --------------------------------------------------------------------------- #
def _patch_ssh_externals(monkeypatch):
    """Stub the ssh_cmd externals so it exits cleanly via the public-endpoint path."""
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/" + x)
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: 0)
    monkeypatch.setattr(
        cli.sdk, "ssh_endpoint",
        lambda name, team=None: {"public_ip": "1.2.3.4", "node_port": 30022, "node": "wuji-0"},
    )
    # Guard: real status must never be reached in these CLI tests.
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("status must not be called")),
    )


def test_ssh_default_calls_wait_once(monkeypatch):
    _patch_ssh_externals(monkeypatch)
    seen = {"n": 0}

    def fake_wait(name, *, team, pod=None, worker=None, timeout=120):
        seen["n"] += 1
        seen["args"] = (name, team, pod, worker)

    monkeypatch.setattr(cli, "_wait_pod_running", fake_wait)
    result = runner.invoke(cli.app, ["ssh", "j", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen["n"] == 1
    assert seen["args"] == ("j", "teamx", None, None)


def test_ssh_no_wait_skips_wait(monkeypatch):
    _patch_ssh_externals(monkeypatch)
    seen = {"n": 0}
    monkeypatch.setattr(
        cli, "_wait_pod_running",
        lambda *a, **k: seen.__setitem__("n", seen["n"] + 1),
    )
    result = runner.invoke(cli.app, ["ssh", "j", "-t", "teamx", "--no-wait"])
    assert result.exit_code == 0, result.output
    assert seen["n"] == 0


def test_ssh_default_passes_worker_to_wait(monkeypatch):
    """--worker N flows through to _wait_pod_running (and worker endpoint is used)."""
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/" + x)
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: 0)
    monkeypatch.setattr(
        cli.sdk, "worker_ssh_endpoint",
        lambda name, worker, team=None: {
            "public_ip": "1.2.3.4", "node_port": 30099, "node": "wuji-1",
        },
    )
    seen = {}

    def fake_wait(name, *, team, pod=None, worker=None, timeout=120):
        seen.update(name=name, team=team, pod=pod, worker=worker)

    monkeypatch.setattr(cli, "_wait_pod_running", fake_wait)
    result = runner.invoke(cli.app, ["ssh", "j", "-t", "teamx", "--worker", "2"])
    assert result.exit_code == 0, result.output
    assert seen == {"name": "j", "team": "teamx", "pod": None, "worker": 2}


def test_ssh_default_passes_pod_to_wait(monkeypatch):
    """--pod flows through to _wait_pod_running."""
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/" + x)
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: 0)
    # --pod path: no public endpoint, falls through to port-forward. Stub Popen + status.
    monkeypatch.setattr(
        cli.sdk, "ssh_endpoint",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("pod path must not use ssh_endpoint")),
    )

    class _FakePopen:
        def __init__(self, *a, **k):
            pass

        def terminate(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    seen = {}

    def fake_wait(name, *, team, pod=None, worker=None, timeout=120):
        seen.update(name=name, team=team, pod=pod, worker=worker)

    monkeypatch.setattr(cli, "_wait_pod_running", fake_wait)
    # local import of time inside ssh_cmd -> real time.sleep(2) in port-forward path;
    # patch module time.sleep to avoid the 2s wait.
    monkeypatch.setattr(cli.time, "sleep", lambda *_a, **_k: None)
    result = runner.invoke(cli.app, ["ssh", "j", "-t", "teamx", "--pod", "j-worker-0"])
    assert result.exit_code == 0, result.output
    assert seen == {"name": "j", "team": "teamx", "pod": "j-worker-0", "worker": None}


def test_ssh_no_ssh_binary_errors_before_wait(monkeypatch):
    """No ssh binary -> error out before ever waiting (wait must not run)."""
    monkeypatch.setattr(shutil, "which", lambda x: None)
    called = {"n": 0}
    monkeypatch.setattr(
        cli, "_wait_pod_running",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1),
    )
    result = runner.invoke(cli.app, ["ssh", "j", "-t", "teamx"])
    assert result.exit_code == 1
    assert called["n"] == 0
