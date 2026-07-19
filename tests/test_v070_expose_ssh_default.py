"""v0.7.0 regression: `volcano dev` now defaults ``--expose-ssh`` to True.

The source change (volcano/cli.py):

    dev(...):  expose_ssh: bool = typer.Option(True, "--expose-ssh/--no-expose-ssh", ...)

``train`` is unchanged (still defaults False). ``dev`` forwards the resolved value
to ``_submit_impl(expose_ssh=...)`` which forwards it to ``sdk.submit``.

These tests pin the new default end-to-end and, critically, keep the suite FAST:
the expose path in ``_submit_impl`` polls ``sdk.ssh_endpoint`` in a
``for _ in range(15): ... time.sleep(2)`` loop. We therefore

* monkeypatch ``cli.time.sleep`` to a counter (never really sleeps), and
* stub ``cli.sdk.ssh_endpoint`` / ``cli.sdk.worker_ssh_endpoint`` to return a
  resolved endpoint so the loop breaks on the first iteration, and
* stub ``cli.sdk.submit`` to capture kwargs and return a dict carrying
  ``_ssh_node_port`` (mirrors a real expose submit).

Only tests/ is touched — no source edits.
"""

import pytest

import volcano.cli as cli
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_interactive_password_prompt(monkeypatch):
    # expose_ssh=True implies ssh -> without an explicit --ssh-password these
    # tests would otherwise hit a real (blocking) getpass() call, since SSH is
    # password-only with no local-key auto-detect (punchlist). Stub it so the
    # existing tests (written before that change) keep exercising the same
    # expose_ssh/submit-kwarg wiring without needing every call site touched.
    monkeypatch.setattr(cli.getpass, "getpass", lambda *_a, **_k: "test-password")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _patch_submit_capture(monkeypatch, box, *, ret=None):
    """Stub sdk.submit: capture kwargs into ``box``, return ``ret`` (default {})."""
    def fake_submit(**kw):
        box.update(kw)
        return {} if ret is None else ret
    monkeypatch.setattr(cli.sdk, "submit", fake_submit)


def _patch_no_sleep(monkeypatch):
    """Replace cli.time.sleep with a counter so the expose poll never really sleeps."""
    sleeps = {"n": 0}
    monkeypatch.setattr(
        cli.time, "sleep",
        lambda *_a, **_k: sleeps.__setitem__("n", sleeps["n"] + 1),
    )
    return sleeps


def _patch_endpoints_resolved(monkeypatch):
    """Make ssh_endpoint / worker_ssh_endpoint resolve immediately (loop breaks)."""
    ep = {"public_ip": "1.2.3.4", "node_port": 30022, "node": "wuji-0"}
    monkeypatch.setattr(cli.sdk, "ssh_endpoint", lambda name, team=None: dict(ep))
    monkeypatch.setattr(
        cli.sdk, "worker_ssh_endpoint",
        lambda name, worker, team=None: dict(ep, node_port=30022 + worker),
    )
    # Guard: a real cluster client must never be built in these unit tests.
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )


# --------------------------------------------------------------------------- #
# ① dev (no ssh flag) -> expose_ssh forwarded as True, end-to-end into sdk.submit
# --------------------------------------------------------------------------- #
def test_dev_default_expose_ssh_true_reaches_submit(monkeypatch):
    box = {}
    sleeps = _patch_no_sleep(monkeypatch)
    _patch_submit_capture(monkeypatch, box, ret={"_ssh_node_port": 30022})
    _patch_endpoints_resolved(monkeypatch)

    result = runner.invoke(cli.app, ["dev", "-n", "d", "-t", "teamx", "-i", "img"])
    assert result.exit_code == 0, result.output
    # New default: dev turns on SSH expose without any flag.
    assert box["expose_ssh"] is True
    # The endpoint resolved on the first poll, so we never actually slept.
    assert sleeps["n"] == 0
    # And the CLI printed the public-direct-connect hint (expose branch ran).
    assert "SSH 公网直连就绪" in result.output
    assert "ssh -p 30022 root@1.2.3.4" in result.output


# --------------------------------------------------------------------------- #
# ② dev --no-expose-ssh -> expose_ssh forwarded as False, no expose branch/poll
# --------------------------------------------------------------------------- #
def test_dev_no_expose_ssh_forwards_false(monkeypatch):
    box = {}
    sleeps = _patch_no_sleep(monkeypatch)
    _patch_submit_capture(monkeypatch, box)  # returns {} -> no _ssh_node_port
    # ssh_endpoint must NOT be consulted when expose is off.
    monkeypatch.setattr(
        cli.sdk, "ssh_endpoint",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ssh_endpoint must not run")),
    )

    result = runner.invoke(
        cli.app, ["dev", "-n", "d", "-t", "teamx", "-i", "img", "--no-expose-ssh"]
    )
    assert result.exit_code == 0, result.output
    assert box["expose_ssh"] is False
    assert sleeps["n"] == 0
    assert "SSH 公网直连就绪" not in result.output


# --------------------------------------------------------------------------- #
# ③ train (no ssh flag) -> expose_ssh default is STILL False (unchanged)
# --------------------------------------------------------------------------- #
def test_train_default_expose_ssh_still_false(monkeypatch):
    box = {}
    _patch_no_sleep(monkeypatch)
    _patch_submit_capture(monkeypatch, box)
    # If train had regressed to expose-by-default, the (unstubbed) endpoint poll
    # would run; assert the flag directly instead.
    monkeypatch.setattr(
        cli.sdk, "ssh_endpoint",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("train must not expose by default")),
    )

    result = runner.invoke(
        cli.app, ["train", "-n", "j", "-t", "teamx", "-i", "img", "--cmd", "c"]
    )
    assert result.exit_code == 0, result.output
    assert box["expose_ssh"] is False


# --------------------------------------------------------------------------- #
# ④ train --expose-ssh -> opt-in still works (flag plumbs through)
# --------------------------------------------------------------------------- #
def test_train_explicit_expose_ssh_true(monkeypatch):
    box = {}
    _patch_no_sleep(monkeypatch)
    _patch_submit_capture(monkeypatch, box, ret={"_ssh_node_port": 30022})
    _patch_endpoints_resolved(monkeypatch)

    result = runner.invoke(
        cli.app,
        ["train", "-n", "j", "-t", "teamx", "-i", "img", "--cmd", "c", "--expose-ssh"],
    )
    assert result.exit_code == 0, result.output
    assert box["expose_ssh"] is True


# --------------------------------------------------------------------------- #
# ⑤ default expose is idempotent w.r.t. custom command / gpus (still exposes)
# --------------------------------------------------------------------------- #
def test_dev_default_expose_with_custom_command(monkeypatch):
    box = {}
    _patch_no_sleep(monkeypatch)
    _patch_submit_capture(monkeypatch, box, ret={"_ssh_node_port": 30022})
    _patch_endpoints_resolved(monkeypatch)

    result = runner.invoke(
        cli.app,
        ["dev", "-n", "d", "-t", "teamx", "-i", "img", "--", "sleep", "100"],
    )
    assert result.exit_code == 0, result.output
    assert box["expose_ssh"] is True
    assert box["command"] == ["sleep", "100"]


# --------------------------------------------------------------------------- #
# ⑥ adversarial: endpoint stays unresolved -> loop must terminate (bounded),
#    not spin forever. With a no-op sleep it should exhaust range(15) at most.
# --------------------------------------------------------------------------- #
def test_dev_default_expose_unresolved_endpoint_is_bounded(monkeypatch):
    box = {}
    sleeps = _patch_no_sleep(monkeypatch)
    _patch_submit_capture(monkeypatch, box, ret={"_ssh_node_port": 30022})
    # Never resolves a public_ip AND never reports a node -> the loop runs its
    # full bound (range(15)) then falls through to the "IP 待解析" hint.
    monkeypatch.setattr(cli.sdk, "ssh_endpoint", lambda name, team=None: None)
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )

    result = runner.invoke(cli.app, ["dev", "-n", "d", "-t", "teamx", "-i", "img"])
    assert result.exit_code == 0, result.output
    assert box["expose_ssh"] is True
    # Bounded: single-node expose loop is range(15) -> at most 15 sleeps.
    assert sleeps["n"] <= 15
    # Falls back to the NodePort-assigned-but-IP-pending guidance.
    assert "NodePort 30022 已分配" in result.output


# --------------------------------------------------------------------------- #
# ⑦ dev --dry-run must not touch submit/expose at all (fast, no poll)
# --------------------------------------------------------------------------- #
def test_dev_dry_run_skips_expose(monkeypatch):
    _patch_no_sleep(monkeypatch)
    monkeypatch.setattr(
        cli.sdk, "submit",
        lambda **kw: (_ for _ in ()).throw(AssertionError("no submit on dry-run")),
    )
    monkeypatch.setattr(
        cli.sdk, "ssh_endpoint",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no expose poll on dry-run")),
    )
    result = runner.invoke(
        cli.app, ["dev", "-n", "d", "-t", "teamx", "-i", "img", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "wuji.io/kind: dev" in result.output
