"""Regression tests for v0.4.0 changes (relative to v0.3.0).

Covers the 5 documented deltas — all mocked, no real cluster / kubeconfig:

  1. `vc` console-script alias: pyproject declares BOTH `volcano` and `vc`
     pointing at `volcano.cli:main`, and that target resolves to a callable.
  2. Short options `-n` (--name), `-d` (--data), `-w` (--worker) bind to the
     right options on train/dev (name/data) and logs/exec/ssh/forward/save
     (worker).
  3. `list -A/--all-namespaces` (also `-a`): sdk.list_jobs(all_namespaces=True)
     takes the cluster-custom-object branch and every returned row carries a
     `namespace`; the CLI adds an NS column and forwards all_namespaces=True,
     while a plain `list` stays single-ns.
  4. `login`: a valid kubeconfig is written to a tmp --dest and context/ns are
     printed; an existing dest is backed up first; invalid content (missing
     clusters/contexts/users or non-YAML) exits non-zero and does NOT overwrite.
  5. Default queue for train/dev is "default" (was "shared" in v0.3.0).

Only tests/ is touched; source/config untouched.
"""

import importlib
import pathlib
import subprocess

import pytest
import yaml
from typer.testing import CliRunner

import volcano.cli as cli
import volcano.sdk as sdk

runner = CliRunner()

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _capture_submit(monkeypatch):
    """Mock sdk.submit to capture kwargs (returns {} like a real submit)."""
    captured = {}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: captured.update(kw) or {})
    return captured


def _guard_no_kube(monkeypatch):
    """Fail loudly if any test in this module touches a real cluster."""
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )


_VALID_KUBECONFIG = {
    "apiVersion": "v1",
    "kind": "Config",
    "current-context": "teamx-ctx",
    "clusters": [{"name": "c1", "cluster": {"server": "https://1.2.3.4:6443"}}],
    "contexts": [
        {"name": "teamx-ctx", "context": {"cluster": "c1", "user": "u1", "namespace": "teamx"}}
    ],
    "users": [{"name": "u1", "user": {"token": "deadbeef"}}],
}


# --------------------------------------------------------------------------- #
# 1. vc alias
# --------------------------------------------------------------------------- #
def test_pyproject_declares_both_console_scripts():
    try:
        import tomllib
    except ModuleNotFoundError:  # py<3.11
        pytest.skip("tomllib unavailable")
    data = tomllib.loads((_PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts.get("volcano") == "volcano.cli:main"
    assert scripts.get("vc") == "volcano.cli:main", "v0.4.0 must add the `vc` alias"


def test_vc_entrypoint_target_is_callable():
    # Resolve "volcano.cli:main" the way an installed console script would.
    mod_name, func_name = "volcano.cli:main".split(":")
    func = getattr(importlib.import_module(mod_name), func_name)
    assert callable(func)


# --------------------------------------------------------------------------- #
# 2. short options -n / -d / -w
# --------------------------------------------------------------------------- #
def test_train_short_name_and_data(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(
        cli.app,
        ["train", "-n", "jshort", "-t", "teamx", "-i", "img", "-d", "sub/dir", "--cmd", "c"],
    )
    assert result.exit_code == 0, result.output
    assert captured["name"] == "jshort"
    assert captured["data"] == "sub/dir"


def test_dev_short_name_and_data(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(
        cli.app,
        ["dev", "-n", "dshort", "-t", "teamx", "-i", "img", "-d", "ckpts",
         "--no-expose-ssh"],
    )
    assert result.exit_code == 0, result.output
    assert captured["name"] == "dshort"
    assert captured["data"] == "ckpts"


def test_logs_short_worker(monkeypatch):
    seen = {}

    def fake_logs(name, team=None, follow=False, tail=200, pod=None, worker=None):
        seen.update(name=name, worker=worker)
        return "log-body"

    monkeypatch.setattr(cli.sdk, "logs", fake_logs)
    result = runner.invoke(cli.app, ["logs", "myjob", "-t", "teamx", "-w", "3"])
    assert result.exit_code == 0, result.output
    assert seen == {"name": "myjob", "worker": 3}


def test_save_short_worker(monkeypatch):
    seen = {}

    def fake_save(name, tag=None, team=None, worker=None, container=None, wait=True, timeout=1800):
        seen.update(name=name, worker=worker, tag=tag)
        return {"image": "reg/wuji-rl/teamx/myenv:v1", "request": "req-1", "status": "Succeeded"}

    monkeypatch.setattr(cli.sdk, "save_image", fake_save)
    # team short-circuits current_namespace, so no kube contact needed.
    result = runner.invoke(
        cli.app, ["save", "myjob", "--tag", "myenv:v1", "-t", "teamx", "-w", "2"]
    )
    assert result.exit_code == 0, result.output
    assert seen["worker"] == 2
    assert seen["name"] == "myjob"


def test_exec_short_worker(monkeypatch):
    # -w resolves target pod to <name>-worker-N without hitting sdk.status.
    argv_box = {}
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/kubectl")
    monkeypatch.setattr(subprocess, "call", lambda argv: argv_box.update(argv=argv) or 0)
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("status must not be called with -w")),
    )
    result = runner.invoke(cli.app, ["exec", "myjob", "-t", "teamx", "-w", "5", "--", "echo", "hi"])
    assert result.exit_code == 0, result.output
    assert "myjob-worker-5" in argv_box["argv"]


def test_ssh_short_worker(monkeypatch):
    seen = {}

    def fake_endpoint(name, worker, team=None):
        seen.update(name=name, worker=worker)
        return {"public_ip": "8.8.8.8", "node_port": 31000, "node": "wuji-0"}

    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/ssh")
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: 0)
    monkeypatch.setattr(cli.sdk, "worker_ssh_endpoint", fake_endpoint)
    result = runner.invoke(cli.app, ["ssh", "myjob", "-t", "teamx", "-w", "4"])
    assert result.exit_code == 0, result.output
    assert seen == {"name": "myjob", "worker": 4}


def test_forward_short_worker(monkeypatch):
    argv_box = {}
    monkeypatch.setattr("shutil.which", lambda _x: "/usr/bin/kubectl")
    monkeypatch.setattr(subprocess, "call", lambda argv: argv_box.update(argv=argv) or 0)
    monkeypatch.setattr(
        cli.sdk, "status",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("status must not be called with -w")),
    )
    result = runner.invoke(cli.app, ["forward", "myjob", "6006", "-t", "teamx", "-w", "1"])
    assert result.exit_code == 0, result.output
    assert "myjob-worker-1" in argv_box["argv"]


# --------------------------------------------------------------------------- #
# 3a. sdk.list_jobs(all_namespaces=True)
# --------------------------------------------------------------------------- #
def _ns_item(name, ns, kind=None):
    labels = {"app.kubernetes.io/managed-by": "wuji"}
    if kind:
        labels["wuji.io/kind"] = kind
    return {
        "metadata": {
            "name": name, "namespace": ns, "labels": labels,
            "creationTimestamp": "2026-01-01T00:00:00Z",
        },
        "spec": {"queue": "default", "tasks": [{"replicas": 1, "template": {"spec": {
            "containers": [{"resources": {"limits": {"nvidia.com/gpu": "8"}}}]}}}]},
        "status": {"state": {"phase": "Running"}},
    }


@pytest.fixture
def dual_custom(monkeypatch):
    """Custom client recording which list method the SDK calls."""
    calls = {"cluster": 0, "namespaced": 0}
    box = {"items": []}

    class Custom:
        def list_cluster_custom_object(self, **kw):
            calls["cluster"] += 1
            return {"items": box["items"]}

        def list_namespaced_custom_object(self, **kw):
            calls["namespaced"] += 1
            calls["ns"] = kw.get("namespace")
            return {"items": box["items"]}

    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, Custom()))
    return calls, box


def test_list_jobs_all_namespaces_uses_cluster_object(dual_custom, monkeypatch):
    calls, box = dual_custom
    box["items"] = [_ns_item("a", "team-a", kind="train"), _ns_item("b", "team-b", kind="dev")]
    monkeypatch.setattr(sdk, "_require_admin", lambda: None)
    # all_namespaces must NOT consult current_namespace (admin/edge view).
    monkeypatch.setattr(
        sdk, "current_namespace",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no current_namespace in -A mode")),
    )
    out = sdk.list_jobs(all_namespaces=True)
    assert calls["cluster"] == 1 and calls["namespaced"] == 0
    assert {j["name"]: j["namespace"] for j in out} == {"a": "team-a", "b": "team-b"}


def test_list_jobs_all_namespaces_honors_kind_filter(dual_custom, monkeypatch):
    calls, box = dual_custom
    box["items"] = [_ns_item("a", "team-a", kind="train"), _ns_item("b", "team-b", kind="dev")]
    monkeypatch.setattr(sdk, "_require_admin", lambda: None)
    monkeypatch.setattr(sdk, "current_namespace", lambda *a, **k: "should-not-be-used")
    out = sdk.list_jobs(all_namespaces=True, kind="train")
    assert [j["name"] for j in out] == ["a"]
    assert out[0]["namespace"] == "team-a"


def test_list_jobs_all_namespaces_requires_admin(dual_custom, monkeypatch):
    # A Job's pod template carries full env/command — same sensitivity as a raw
    # Pod, so -A must be genuinely admin-gated, not handed to every registered
    # user via volcano-cluster-viewer (see sdk._CLUSTER_VIEWER_RULES comment).
    calls, box = dual_custom
    box["items"] = [_ns_item("a", "team-a", kind="train")]

    def deny():
        raise sdk.WujiError("该命令需要管理员 kubeconfig")

    monkeypatch.setattr(sdk, "_require_admin", deny)
    with pytest.raises(sdk.WujiError, match="管理员"):
        sdk.list_jobs(all_namespaces=True)
    assert calls["cluster"] == 0  # denied before ever touching the cluster-wide list


def test_list_jobs_single_ns_does_not_require_admin(dual_custom, monkeypatch):
    # Per-own-namespace listing (the common case) must NOT be admin-gated.
    calls, box = dual_custom
    box["items"] = [_ns_item("a", "teamx", kind="train")]

    def boom():
        raise AssertionError("_require_admin must not run for single-namespace list")

    monkeypatch.setattr(sdk, "_require_admin", boom)
    monkeypatch.setattr(sdk, "current_namespace", lambda *a, **k: "teamx")
    out = sdk.list_jobs(team="teamx")
    assert calls["namespaced"] == 1
    assert [j["name"] for j in out] == ["a"]


def test_list_jobs_single_ns_uses_namespaced_object(dual_custom, monkeypatch):
    calls, box = dual_custom
    box["items"] = [_ns_item("a", "teamx")]
    monkeypatch.setattr(sdk, "current_namespace", lambda team=None: "teamx")
    out = sdk.list_jobs(team="teamx")
    assert calls["namespaced"] == 1 and calls["cluster"] == 0
    assert calls["ns"] == "teamx"
    assert out[0]["namespace"] == "teamx"


# --------------------------------------------------------------------------- #
# 3b. CLI `list -A` / `-a`
# --------------------------------------------------------------------------- #
def _mock_list_jobs(monkeypatch, rows):
    seen = {}

    def fake(team=None, kind=None, all_namespaces=False):
        seen.update(team=team, kind=kind, all_namespaces=all_namespaces)
        return rows

    monkeypatch.setattr(cli.sdk, "list_jobs", fake)
    return seen


def test_cli_list_all_namespaces_flag(monkeypatch):
    _guard_no_kube(monkeypatch)
    seen = _mock_list_jobs(monkeypatch, [
        {"namespace": "nsa", "name": "j1", "kind": "train", "phase": "Running",
         "queue": "default", "gpus": 8, "age": "1m"},
    ])
    result = runner.invoke(cli.app, ["list", "-A"])
    assert result.exit_code == 0, result.output
    assert seen["all_namespaces"] is True
    assert "NS" in result.output       # extra namespace column header
    assert "nsa" in result.output      # namespace value shown


def test_cli_list_lowercase_a_alias(monkeypatch):
    _guard_no_kube(monkeypatch)
    seen = _mock_list_jobs(monkeypatch, [])
    result = runner.invoke(cli.app, ["list", "-a"])
    assert result.exit_code == 0, result.output
    assert seen["all_namespaces"] is True


def test_cli_list_without_flag_is_single_ns(monkeypatch):
    _guard_no_kube(monkeypatch)
    seen = _mock_list_jobs(monkeypatch, [
        {"namespace": "teamx", "name": "j1", "kind": "train", "phase": "Running",
         "queue": "default", "gpus": 8, "age": "1m"},
    ])
    # team short-circuits current_namespace for the title, so no kube contact.
    result = runner.invoke(cli.app, ["list", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen["all_namespaces"] is False
    # single-ns table has no NS column header
    assert "\nNS " not in result.output and "NS\n" not in result.output


# --------------------------------------------------------------------------- #
# 4. login
# --------------------------------------------------------------------------- #
def test_login_valid_writes_dest_and_prints_ctx(tmp_path):
    src = tmp_path / "team.kubeconfig"
    src.write_text(yaml.safe_dump(_VALID_KUBECONFIG), encoding="utf-8")
    dest = tmp_path / "kube" / "config"
    result = runner.invoke(cli.app, ["login", "-f", str(src), "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    assert dest.exists()
    # content copied verbatim
    written = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert written["current-context"] == "teamx-ctx"
    assert "teamx-ctx" in result.output
    assert "teamx" in result.output  # namespace printed


def test_login_via_stdin_paste(tmp_path):
    dest = tmp_path / "config"
    result = runner.invoke(
        cli.app, ["login", "--dest", str(dest)],
        input=yaml.safe_dump(_VALID_KUBECONFIG),
    )
    assert result.exit_code == 0, result.output
    assert dest.exists()
    assert yaml.safe_load(dest.read_text(encoding="utf-8"))["kind"] == "Config"


def test_login_backs_up_existing_dest(tmp_path):
    dest = tmp_path / "config"
    dest.write_text("OLD-CONFIG-CONTENT\n", encoding="utf-8")
    src = tmp_path / "new.kubeconfig"
    src.write_text(yaml.safe_dump(_VALID_KUBECONFIG), encoding="utf-8")
    result = runner.invoke(cli.app, ["login", "-f", str(src), "--dest", str(dest)])
    assert result.exit_code == 0, result.output
    baks = list(tmp_path.glob("config.bak.*"))
    assert len(baks) == 1, f"expected one backup, got {baks}"
    assert baks[0].read_text(encoding="utf-8") == "OLD-CONFIG-CONTENT\n"  # old preserved
    # dest now holds the new config
    assert yaml.safe_load(dest.read_text(encoding="utf-8"))["current-context"] == "teamx-ctx"
    assert "已备份" in result.output


def test_login_force_skips_backup(tmp_path):
    dest = tmp_path / "config"
    dest.write_text("OLD\n", encoding="utf-8")
    src = tmp_path / "new.kubeconfig"
    src.write_text(yaml.safe_dump(_VALID_KUBECONFIG), encoding="utf-8")
    result = runner.invoke(cli.app, ["login", "-f", str(src), "--dest", str(dest), "--force"])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("config.bak.*")) == []
    assert "已备份" not in result.output


@pytest.mark.parametrize("bad", [
    # missing users key
    {"apiVersion": "v1", "clusters": [], "contexts": []},
    # missing contexts + users
    {"clusters": []},
    # a YAML scalar, not a mapping
    "just a string",
])
def test_login_invalid_kubeconfig_exits_nonzero_no_overwrite(tmp_path, bad):
    dest = tmp_path / "config"
    dest.write_text("PRECIOUS\n", encoding="utf-8")
    src = tmp_path / "bad.yaml"
    src.write_text(yaml.safe_dump(bad) if not isinstance(bad, str) else bad, encoding="utf-8")
    result = runner.invoke(cli.app, ["login", "-f", str(src), "--dest", str(dest)])
    assert result.exit_code != 0, result.output
    # existing dest untouched, no backup made
    assert dest.read_text(encoding="utf-8") == "PRECIOUS\n"
    assert list(tmp_path.glob("config.bak.*")) == []


def test_login_non_yaml_content_exits_nonzero(tmp_path):
    dest = tmp_path / "config"
    src = tmp_path / "bad.yaml"
    # Unbalanced brackets -> YAMLError, not a valid document.
    src.write_text("clusters: [unclosed\n:::", encoding="utf-8")
    result = runner.invoke(cli.app, ["login", "-f", str(src), "--dest", str(dest)])
    assert result.exit_code != 0, result.output
    assert not dest.exists()


def test_login_missing_file_exits_nonzero(tmp_path):
    dest = tmp_path / "config"
    result = runner.invoke(
        cli.app, ["login", "-f", str(tmp_path / "does-not-exist"), "--dest", str(dest)]
    )
    assert result.exit_code != 0, result.output
    assert not dest.exists()


# --------------------------------------------------------------------------- #
# 5. default queue == "default" (was "shared" in v0.3.0)
# --------------------------------------------------------------------------- #
def test_train_default_queue_is_default(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(cli.app, ["train", "-n", "j", "-t", "teamx", "-i", "img", "--cmd", "c"])
    assert result.exit_code == 0, result.output
    assert captured["queue"] == "default"


def test_dev_default_queue_is_default(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_submit(monkeypatch)
    result = runner.invoke(
        cli.app, ["dev", "-n", "d", "-t", "teamx", "-i", "img", "--no-expose-ssh"]
    )
    assert result.exit_code == 0, result.output
    assert captured["queue"] == "default"


def test_train_dry_run_default_queue_in_yaml(monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "submit",
        lambda **kw: (_ for _ in ()).throw(AssertionError("no submit on dry-run")),
    )
    result = runner.invoke(
        cli.app,
        ["train", "-n", "dj", "-t", "teamx", "-i", "img", "--cmd", "python t.py", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load(result.output)
    assert manifest["spec"]["queue"] == "default"
