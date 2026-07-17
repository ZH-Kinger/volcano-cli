"""Regression tests for the 4 rename-era UX fixes (wuji -> volcano).

Covered:
  1. kube.current_namespace: raises WujiError when no team / no kubeconfig
     context ns / no in-cluster SA (no more silent "default"); team wins;
     kubeconfig context ns still resolves.
  2. cli `data ls`: helper Pod image is ubuntu:24.04; Pending-timeout message
     (image/schedule problem, not a path problem); Failed + log containing
     "No such file"/"cannot access" -> "路径不存在"; other Failed -> generic.
  3. cli `queue`: prints the "only counts Volcano-scheduled" hint line.
  4. cli `list`: empty result prints the "native kubectl Job not listed" hint.

Only tests/ is touched. Clients / current_namespace are mocked; no real
cluster or kubeconfig is contacted.
"""

import builtins
import types

import pytest
from typer.testing import CliRunner

import volcano.cli as cli
import volcano.kube as kube
from volcano.kube import WujiError

runner = CliRunner()

SA_NS_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


# --------------------------------------------------------------------------- #
# fix 1: current_namespace no longer silently returns "default"
# --------------------------------------------------------------------------- #
def _block_sa_file(monkeypatch):
    """Make reading the in-cluster SA namespace file fail, deterministically."""
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path) == SA_NS_PATH:
            raise OSError("no in-cluster SA in test")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_current_namespace_team_wins(monkeypatch):
    # team short-circuits: kube config must not even be consulted.
    monkeypatch.setattr(
        kube.config, "list_kube_config_contexts",
        lambda: (_ for _ in ()).throw(AssertionError("must not touch kubeconfig when team given")),
    )
    assert kube.current_namespace("teamx") == "teamx"


def test_current_namespace_uses_kubeconfig_context_ns(monkeypatch):
    monkeypatch.setattr(
        kube.config, "list_kube_config_contexts",
        lambda: ([], {"name": "ctx", "context": {"namespace": "ctxns"}}),
    )
    assert kube.current_namespace() == "ctxns"


def test_current_namespace_raises_when_no_info(monkeypatch):
    # No team, kubeconfig context carries no namespace, no in-cluster SA file.
    monkeypatch.setattr(
        kube.config, "list_kube_config_contexts",
        lambda: ([], {"name": "ctx", "context": {}}),
    )
    _block_sa_file(monkeypatch)
    with pytest.raises(WujiError) as ei:
        kube.current_namespace()
    assert "无法确定团队 namespace" in str(ei.value)
    # must NOT have fallen back to "default"
    assert "default" not in str(ei.value).split("namespace")[0]


def test_current_namespace_raises_when_kubeconfig_missing(monkeypatch):
    # list_kube_config_contexts raising (no kubeconfig) is swallowed, then no SA -> raise.
    monkeypatch.setattr(
        kube.config, "list_kube_config_contexts",
        lambda: (_ for _ in ()).throw(kube.ConfigException("no kubeconfig")),
    )
    _block_sa_file(monkeypatch)
    with pytest.raises(WujiError):
        kube.current_namespace()


# --------------------------------------------------------------------------- #
# fix 2: `data ls` helper Pod
# --------------------------------------------------------------------------- #
def _fake_pod(phase):
    return types.SimpleNamespace(status=types.SimpleNamespace(phase=phase))


class _FakeCore:
    """Minimal CoreV1Api stand-in recording create/read/log/delete calls."""

    def __init__(self, phase, log=""):
        self.phase = phase
        self.log = log
        self.created = None
        self.deleted = None

    def create_namespaced_pod(self, namespace=None, body=None):
        self.created = {"namespace": namespace, "body": body}
        return None

    def read_namespaced_pod(self, name=None, namespace=None):
        return _fake_pod(self.phase)

    def read_namespaced_pod_log(self, name=None, namespace=None):
        return self.log

    def delete_namespaced_pod(self, name=None, namespace=None):
        self.deleted = {"name": name, "namespace": namespace}
        return None


def _install_core(monkeypatch, core):
    monkeypatch.setattr(cli, "load_clients", lambda: (core, None, None))
    # data_ls sleeps between polls; keep tests instant if a poll ever happens.
    monkeypatch.setattr(cli.time, "sleep", lambda *_a, **_k: None)


def test_data_ls_helper_pod_image_is_ubuntu(monkeypatch):
    core = _FakeCore("Succeeded", log="total 0\ndrwxr-xr-x foo\n")
    _install_core(monkeypatch, core)
    result = runner.invoke(cli.app, ["data", "ls", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    container = core.created["body"]["spec"]["containers"][0]
    assert container["image"] == "ubuntu:24.04"
    # non-shared -> team NAS PVC + /workspace mount
    assert core.created["body"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"] == "teamx-nas"
    assert "total 0" in result.output
    # helper Pod is always cleaned up
    assert core.deleted is not None


def test_data_ls_shared_uses_datasets_mount(monkeypatch):
    core = _FakeCore("Succeeded", log="ok")
    _install_core(monkeypatch, core)
    result = runner.invoke(cli.app, ["data", "ls", "-t", "teamx", "--shared"])
    assert result.exit_code == 0, result.output
    container = core.created["body"]["spec"]["containers"][0]
    assert "/datasets" in container["command"][-1]
    assert core.created["body"]["spec"]["volumes"][0]["persistentVolumeClaim"]["claimName"] == "shared-datasets"


def test_data_ls_pending_timeout_message(monkeypatch):
    core = _FakeCore("Pending")
    _install_core(monkeypatch, core)
    # --timeout 0 => the poll loop never runs, phase stays Pending, no sleep/wait.
    result = runner.invoke(cli.app, ["data", "ls", "-t", "teamx", "--timeout", "0"])
    assert result.exit_code == 0, result.output
    assert "仍 Pending" in result.output
    assert "镜像" in result.output  # blames image/schedule, not the path
    assert "路径不存在" not in result.output
    assert core.deleted is not None  # still cleaned up


@pytest.mark.parametrize("log", [
    "ls: cannot access '/workspace/nope': No such file or directory",
    "ls: /workspace/nope: No such file or directory",
])
def test_data_ls_failed_path_not_exist(monkeypatch, log):
    core = _FakeCore("Failed", log=log)
    _install_core(monkeypatch, core)
    result = runner.invoke(cli.app, ["data", "ls", "nope", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert "路径不存在" in result.output
    assert "/workspace/nope" in result.output


def test_data_ls_failed_other_is_generic(monkeypatch):
    # Failed with output that is NOT a missing-path signature -> generic error.
    core = _FakeCore("Failed", log="segfault: permission denied")
    _install_core(monkeypatch, core)
    result = runner.invoke(cli.app, ["data", "ls", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert "路径不存在" not in result.output
    assert "执行失败" in result.output


# --------------------------------------------------------------------------- #
# fix 3: `queue` prints the volcano-only accounting hint
# --------------------------------------------------------------------------- #
def test_queue_prints_scheduler_hint(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_queues", lambda: [])
    result = runner.invoke(cli.app, ["queue"])
    assert result.exit_code == 0, result.output
    assert "只统计走 Volcano 调度" in result.output
    assert "volcano top" in result.output


# --------------------------------------------------------------------------- #
# fix 4: `list` empty result prints the native-Job hint
# --------------------------------------------------------------------------- #
def test_list_empty_prints_hint(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_jobs", lambda team=None, kind=None, all_namespaces=False: [])
    result = runner.invoke(cli.app, ["list", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert "(空)注意" in result.output
    assert "原生 Job/Pod" in result.output


def test_list_nonempty_no_hint(monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "list_jobs",
        lambda team=None, kind=None, all_namespaces=False: [
            {"name": "a", "kind": "train", "phase": "Running",
             "queue": "shared", "gpus": 8, "age": "1m"}
        ],
    )
    result = runner.invoke(cli.app, ["list", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert "原生 Job/Pod" not in result.output
