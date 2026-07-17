"""CLI-layer tests (typer.testing.CliRunner) with the sdk mocked out.

Covers: --cmd vs `-- CMD` mutual exclusion, missing-command error, mount_path='/'
rejection, the `-t` short flag for --team, and the post-submit connection hints
(status/logs/exec, and the SSH / expose-ssh variants).
"""

import pytest
from typer.testing import CliRunner

import volcano.cli as cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_kube(monkeypatch):
    # Guard: nothing in these tests should touch a real cluster/kubeconfig.
    monkeypatch.setattr(cli, "load_clients", lambda: (_ for _ in ()).throw(
        AssertionError("load_clients must not be called in CLI unit tests")
    ))


# --------------------------------------------------------------------------- #
# submit: argument validation
# --------------------------------------------------------------------------- #
def test_submit_cmd_and_dashdash_mutually_exclusive(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: called.__setitem__("n", 1))
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "--image", "img", "--cmd", "echo hi", "--", "python", "x.py"],
    )
    assert result.exit_code == 1
    assert "只能给一个" in result.output
    assert called["n"] == 0


def test_submit_requires_a_command(monkeypatch):
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: {})
    result = runner.invoke(cli.app, ["submit", "--name", "j", "--image", "img"])
    assert result.exit_code == 1
    assert "训练命令" in result.output


def test_submit_rejects_root_mount_path(monkeypatch):
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: {})
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "--image", "img", "--mount-path", "/", "--cmd", "echo"],
    )
    assert result.exit_code == 1
    assert "绝对路径" in result.output


def test_submit_rejects_relative_mount_path(monkeypatch):
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: {})
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "--image", "img", "--mount-path", "workspace", "--cmd", "echo"],
    )
    assert result.exit_code == 1


def test_submit_rejects_out_of_range_nodeport(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: called.__setitem__("n", 1))
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "--image", "img", "--cmd", "c",
         "--expose-ssh", "--ssh-nodeport", "12345"],
    )
    assert result.exit_code == 1
    assert "30000-32767" in result.output
    assert called["n"] == 0  # rejected before creating the Job


def test_submit_rejects_nodeport_above_range(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: called.__setitem__("n", 1))
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "--image", "img", "--cmd", "c",
         "--expose-ssh", "--ssh-nodeport", "40000"],
    )
    assert result.exit_code == 1
    assert called["n"] == 0


def test_submit_accepts_in_range_nodeport(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: captured.update(kw) or {"_ssh_node_port": 30000})
    monkeypatch.setattr(cli.sdk, "ssh_endpoint", lambda name, team=None: {
        "public_ip": "1.2.3.4", "node_port": 30000, "node": "wuji-1"
    })
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "-t", "teamx", "--image", "img", "--cmd", "c",
         "--expose-ssh", "--ssh-nodeport", "30000"],
    )
    assert result.exit_code == 0, result.output
    assert captured["ssh_node_port"] == 30000


def test_submit_generated_password_echoed(monkeypatch):
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: {"_ssh_generated_password": "abc123XYZ"})
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "-t", "teamx", "--image", "img", "--cmd", "c", "--ssh"],
    )
    assert result.exit_code == 0, result.output
    assert "abc123XYZ" in result.output


def test_submit_passes_ssh_pubkey_to_sdk(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: captured.update(kw) or {})
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "-t", "teamx", "--image", "img", "--cmd", "c",
         "--ssh-pubkey", "ssh-ed25519 AAAA user@host"],
    )
    assert result.exit_code == 0, result.output
    assert captured["ssh_pubkey"] == "ssh-ed25519 AAAA user@host"


# --------------------------------------------------------------------------- #
# submit: happy path + connection hints
# --------------------------------------------------------------------------- #
def test_submit_prints_connection_hints(monkeypatch):
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: {})
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "myjob", "-t", "teamx", "--image", "img", "--cmd", "python train.py"],
    )
    assert result.exit_code == 0, result.output
    assert "已提交任务: myjob" in result.output
    assert "volcano status myjob --team teamx" in result.output
    assert "volcano logs myjob -f --team teamx" in result.output
    assert "volcano exec myjob --team teamx -- bash" in result.output


def test_submit_passes_expose_ssh_flags_to_sdk(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: captured.update(kw) or {"_ssh_node_port": 31000})
    # avoid the ~30s ssh_endpoint poll loop
    monkeypatch.setattr(cli.sdk, "ssh_endpoint", lambda name, team=None: {
        "public_ip": "1.2.3.4", "node_port": 31000, "node": "wuji-1"
    })
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "-t", "teamx", "--image", "img",
         "--cmd", "c", "--expose-ssh", "--ssh-nodeport", "31000"],
    )
    assert result.exit_code == 0, result.output
    assert captured["expose_ssh"] is True
    assert captured["ssh_node_port"] == 31000
    assert "ssh -p 31000 root@1.2.3.4" in result.output


def test_submit_ssh_hint_when_ssh_flag(monkeypatch):
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: {})
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "-t", "teamx", "--image", "img", "--cmd", "c", "--ssh"],
    )
    assert result.exit_code == 0, result.output
    assert "volcano ssh j --team teamx" in result.output


def test_submit_port_forward_hint(monkeypatch):
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: {})
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "-t", "teamx", "--image", "img", "--cmd", "c", "--port", "6006"],
    )
    assert result.exit_code == 0, result.output
    assert "volcano forward j 6006 --team teamx" in result.output


# --------------------------------------------------------------------------- #
# -t short flag threads through to sdk on read commands
# --------------------------------------------------------------------------- #
def test_list_short_team_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli.sdk, "list_jobs",
        lambda team=None, kind=None, all_namespaces=False: (seen.update(team=team, kind=kind), [])[1],
    )
    result = runner.invoke(cli.app, ["list", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen["team"] == "teamx"


def test_status_short_team_flag(monkeypatch):
    seen = {}

    def fake_status(name, team=None):
        seen["name"] = name
        seen["team"] = team
        return {"name": name, "phase": "Running", "message": "", "pods": [],
                "gpu": 8, "cpu_m": 16000, "mem_gi": 64}

    monkeypatch.setattr(cli.sdk, "status", fake_status)
    result = runner.invoke(cli.app, ["status", "myjob", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen == {"name": "myjob", "team": "teamx"}


def test_kill_short_team_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.sdk, "kill", lambda name, team=None: seen.update(name=name, team=team))
    result = runner.invoke(cli.app, ["kill", "myjob", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen == {"name": "myjob", "team": "teamx"}
    assert "已删除任务: myjob" in result.output


def test_usage_short_team_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.sdk, "usage_by_namespace",
                        lambda team=None: (seen.update(team=team), [])[1])
    result = runner.invoke(cli.app, ["usage", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen["team"] == "teamx"


# --------------------------------------------------------------------------- #
# dry-run builds a manifest without touching the cluster
# --------------------------------------------------------------------------- #
def test_submit_dry_run_prints_yaml(monkeypatch):
    monkeypatch.setattr(cli.sdk, "submit",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("no submit on dry-run")))
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "dj", "-t", "teamx", "--image", "img", "--cmd", "python t.py", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "batch.volcano.sh/v1alpha1" in result.output
    assert "name: dj" in result.output
    assert "namespace: teamx" in result.output


def test_submit_error_exits_nonzero(monkeypatch):
    def boom(**kw):
        raise cli.WujiError("资源冲突(409):同名任务已存在")

    monkeypatch.setattr(cli.sdk, "submit", boom)
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "-t", "teamx", "--image", "img", "--cmd", "c"],
    )
    assert result.exit_code == 1
    assert "资源冲突" in result.output


# --------------------------------------------------------------------------- #
# train / submit / dev  -> kind + dev defaults
# --------------------------------------------------------------------------- #
def _capture_submit(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: captured.update(kw) or {})
    return captured


def test_train_sets_kind_train(monkeypatch):
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(
        cli.app,
        ["train", "--name", "j", "-t", "teamx", "--image", "img", "--cmd", "python x.py"],
    )
    assert result.exit_code == 0, result.output
    assert captured["kind"] == "train"
    assert captured["gpus"] == 8  # train default
    assert captured["nodes"] == 1
    assert "kind=train" in result.output
    assert "已提交任务: j" in result.output


def test_submit_alias_sets_kind_train(monkeypatch):
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(
        cli.app,
        ["submit", "--name", "j", "-t", "teamx", "--image", "img", "--cmd", "c"],
    )
    assert result.exit_code == 0, result.output
    assert captured["kind"] == "train"


def test_dev_sets_kind_and_defaults(monkeypatch):
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(
        cli.app,
        # --no-expose-ssh: this test only cares about kind/gpu/command defaults,
        # not the SSH-expose path (which would otherwise poll for a public IP).
        ["dev", "--name", "d", "-t", "teamx", "--image", "img", "--no-expose-ssh"],
    )
    assert result.exit_code == 0, result.output
    assert captured["kind"] == "dev"
    assert captured["gpus"] == 1  # dev default GPU
    assert captured["nodes"] == 1  # dev pins single node
    assert captured["command"] == "tail -f /dev/null"  # long-lived default
    assert "已提交开发机: d" in result.output
    assert "kind=dev" in result.output


def test_dev_custom_command_overrides_default(monkeypatch):
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(
        cli.app,
        ["dev", "--name", "d", "-t", "teamx", "--image", "img", "--no-expose-ssh",
         "--", "sleep", "100"],
    )
    assert result.exit_code == 0, result.output
    assert captured["command"] == ["sleep", "100"]


def test_dev_explicit_gpus(monkeypatch):
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(
        cli.app,
        ["dev", "--name", "d", "-t", "teamx", "--image", "img", "--gpus", "4",
         "--no-expose-ssh"],
    )
    assert result.exit_code == 0, result.output
    assert captured["gpus"] == 4


def test_train_dry_run_has_kind_label(monkeypatch):
    result = runner.invoke(
        cli.app,
        ["train", "--name", "tj", "-t", "teamx", "--image", "img", "--cmd", "c", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "wuji.io/kind: train" in result.output


def test_dev_dry_run_manifest(monkeypatch):
    result = runner.invoke(
        cli.app,
        ["dev", "--name", "dj", "-t", "teamx", "--image", "img", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "wuji.io/kind: dev" in result.output
    assert "tail -f /dev/null" in result.output
    assert "nvidia.com/gpu: '1'" in result.output  # dev default 1 GPU
    assert "minAvailable: 1" in result.output


# --------------------------------------------------------------------------- #
# list [dev|train] filtering
# --------------------------------------------------------------------------- #
def test_list_rejects_invalid_kind(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cli.sdk, "list_jobs",
                        lambda team=None, kind=None, all_namespaces=False: called.__setitem__("n", 1) or [])
    result = runner.invoke(cli.app, ["list", "foo"])
    assert result.exit_code == 1
    assert called["n"] == 0  # rejected before hitting the SDK


def test_list_dev_passes_kind(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli.sdk, "list_jobs",
        lambda team=None, kind=None, all_namespaces=False: (seen.update(team=team, kind=kind), [])[1],
    )
    result = runner.invoke(cli.app, ["list", "dev", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen == {"team": "teamx", "kind": "dev"}


def test_list_train_passes_kind(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli.sdk, "list_jobs",
        lambda team=None, kind=None, all_namespaces=False: (seen.update(kind=kind), [
            {"name": "a", "kind": "train", "phase": "Running", "queue": "shared",
             "gpus": 8, "age": "t"}
        ])[1],
    )
    # 需带 --team(现在无 team 且无 kubeconfig context ns 时 current_namespace 会明确报错)
    result = runner.invoke(cli.app, ["list", "train", "--team", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen["kind"] == "train"


# --------------------------------------------------------------------------- #
# logs: --all / -f interaction, --worker threading
# --------------------------------------------------------------------------- #
def test_logs_all_with_follow_errors(monkeypatch):
    monkeypatch.setattr(cli.sdk, "status",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no status")))
    monkeypatch.setattr(cli.sdk, "logs",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no logs")))
    result = runner.invoke(cli.app, ["logs", "j", "-t", "teamx", "--all", "-f"])
    assert result.exit_code == 1
    assert "--all" in result.output


def test_logs_all_prints_each_pod(monkeypatch):
    monkeypatch.setattr(cli.sdk, "status", lambda name, team=None: {
        "pods": [{"name": "j-worker-0", "phase": "Running"},
                 {"name": "j-worker-1", "phase": "Running"}]
    })
    monkeypatch.setattr(cli.sdk, "logs",
                        lambda name, team=None, tail=200, pod=None: f"LOG-{pod}")
    result = runner.invoke(cli.app, ["logs", "j", "-t", "teamx", "--all"])
    assert result.exit_code == 0, result.output
    assert "[j-worker-0]" in result.output
    assert "LOG-j-worker-0" in result.output
    assert "LOG-j-worker-1" in result.output


def test_logs_worker_threaded_to_sdk(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli.sdk, "logs",
        lambda name, team=None, follow=False, tail=200, pod=None, worker=None:
        captured.update(name=name, worker=worker, pod=pod) or "out",
    )
    result = runner.invoke(cli.app, ["logs", "j", "-t", "teamx", "--worker", "2"])
    assert result.exit_code == 0, result.output
    assert captured["worker"] == 2
    assert captured["pod"] is None


# --------------------------------------------------------------------------- #
# exec / ssh / forward: --worker resolves <name>-worker-N
# --------------------------------------------------------------------------- #
def test_exec_worker_resolves_pod(monkeypatch):
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/" + x)
    calls = {}
    monkeypatch.setattr(subprocess, "call", lambda argv: (calls.__setitem__("argv", argv), 0)[1])
    # --worker short-circuits pod discovery, so status must NOT be called
    monkeypatch.setattr(cli.sdk, "status",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no status")))
    result = runner.invoke(cli.app, ["exec", "j", "-t", "teamx", "--worker", "2"])
    assert result.exit_code == 0, result.output
    assert "j-worker-2" in calls["argv"]


def test_first_running_pod_worker_resolves():
    assert cli._first_running_pod("j", "teamx", None, 2) == ("teamx", "j-worker-2")


def test_first_running_pod_worker_zero():
    assert cli._first_running_pod("j", "teamx", None, 0) == ("teamx", "j-worker-0")


def test_first_running_pod_pod_wins_over_worker():
    assert cli._first_running_pod("j", "teamx", "explicit-pod", 2) == ("teamx", "explicit-pod")


def test_first_running_pod_default_uses_status(monkeypatch):
    monkeypatch.setattr(cli.sdk, "status", lambda name, team=None: {
        "pods": [{"name": "j-worker-0", "phase": "Running"}]
    })
    assert cli._first_running_pod("j", "teamx", None, None) == ("teamx", "j-worker-0")


def test_ssh_worker_uses_worker_endpoint_public(monkeypatch):
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/" + x)
    calls = {}
    monkeypatch.setattr(subprocess, "call", lambda argv: (calls.__setitem__("argv", argv), 0)[1])
    # --worker must resolve via worker_ssh_endpoint, not the job-level ssh_endpoint
    monkeypatch.setattr(cli.sdk, "worker_ssh_endpoint", lambda name, worker, team=None: {
        "public_ip": "1.2.3.4", "node_port": 30005, "node": "wuji-2",
        "pod": f"{name}-worker-{worker}", "worker": worker,
    })
    monkeypatch.setattr(cli.sdk, "ssh_endpoint",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong endpoint")))
    result = runner.invoke(cli.app, ["ssh", "j", "-t", "teamx", "--worker", "1"])
    assert result.exit_code == 0, result.output
    argv = calls["argv"]
    assert "root@1.2.3.4" in argv
    assert "30005" in argv
