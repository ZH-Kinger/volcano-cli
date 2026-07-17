"""v0.9.0 CLI-layer tests: ``volcano set`` / ``set-queue`` + default-queue routing.

CliRunner drives the commands with ``sdk.*`` mocked (no real cluster / kubeconfig).

* ``set`` (set_cmd): threads --gpu/--cpu/--memory/--pods/--team/--clear into
  ``sdk.set_quota``; prints team-attribution, the quota table, the "no limit"
  hint, the "cleared" line; a ``WujiError`` from the SDK is a friendly non-zero
  exit.
* ``set-queue`` (set_queue_cmd): threads --deserved/--cap/--reclaimable/--delete
  into ``sdk.set_queue``; prints created/updated/deleted lines.
* default-queue routing: ``train``/``dev`` with no ``-q`` call
  ``sdk.resolve_default_queue`` and use its return as the queue (dry-run YAML +
  the value forwarded to ``sdk.submit``); an explicit ``-q`` bypasses resolution.

Only tests/ is touched.
"""

import yaml
from typer.testing import CliRunner

import volcano.cli as cli

runner = CliRunner()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _guard_no_kube(monkeypatch):
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )


def _capture_set_quota(monkeypatch, ret):
    captured = {}

    def fake(namespace, *, gpu=None, cpu=None, memory=None, pods=None, team=None, clear=False):
        captured.update(namespace=namespace, gpu=gpu, cpu=cpu, memory=memory,
                        pods=pods, team=team, clear=clear)
        return ret

    monkeypatch.setattr(cli.sdk, "set_quota", fake)
    return captured


def _capture_set_queue(monkeypatch, ret):
    captured = {}

    def fake(team, *, deserved=None, cap=None, reclaimable=None, delete=False):
        captured.update(team=team, deserved=deserved, cap=cap,
                        reclaimable=reclaimable, delete=delete)
        return ret

    monkeypatch.setattr(cli.sdk, "set_queue", fake)
    return captured


# =========================================================================== #
# volcano set  (per-person ResourceQuota)
# =========================================================================== #
def test_set_gpu_and_team_threaded_to_sdk(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_set_quota(monkeypatch, {
        "namespace": "alice", "team": "redteam",
        "quota": {"namespace": "alice",
                  "hard": {"requests.nvidia.com/gpu": "8"},
                  "used": {"requests.nvidia.com/gpu": "0"}},
    })
    result = runner.invoke(cli.app, ["set", "alice", "--gpu", "8", "--team", "redteam"])
    assert result.exit_code == 0, result.output
    assert captured["namespace"] == "alice"
    assert captured["gpu"] == 8
    assert captured["team"] == "redteam"
    # team attribution line + quota table title
    assert "已归属团队" in result.output and "redteam" in result.output
    assert "资源上限" in result.output


def test_set_short_flags(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_set_quota(monkeypatch, {"namespace": "alice", "team": None, "quota": None})
    result = runner.invoke(
        cli.app, ["set", "alice", "-g", "4", "-c", "32", "-m", "128Gi"]
    )
    assert result.exit_code == 0, result.output
    assert captured["gpu"] == 4 and captured["cpu"] == "32" and captured["memory"] == "128Gi"


def test_set_clear_flag(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_set_quota(monkeypatch, {
        "namespace": "alice", "cleared": True, "team": None, "quota": None,
    })
    result = runner.invoke(cli.app, ["set", "alice", "--clear"])
    assert result.exit_code == 0, result.output
    assert captured["clear"] is True
    assert "已取消" in result.output


def test_set_no_args_reports_no_limit(monkeypatch):
    _guard_no_kube(monkeypatch)
    _capture_set_quota(monkeypatch, {"namespace": "alice", "team": None, "quota": None})
    result = runner.invoke(cli.app, ["set", "alice"])
    assert result.exit_code == 0, result.output
    assert "无资源限制" in result.output
    assert "volcano set alice --gpu 8" in result.output


def test_set_reports_existing_quota_table(monkeypatch):
    _guard_no_kube(monkeypatch)
    _capture_set_quota(monkeypatch, {
        "namespace": "alice", "team": None,
        "quota": {"namespace": "alice",
                  "hard": {"pods": "20"}, "used": {"pods": "3"}},
    })
    result = runner.invoke(cli.app, ["set", "alice"])
    assert result.exit_code == 0, result.output
    assert "资源上限" in result.output
    assert "pods" in result.output


def test_set_pods_flag(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_set_quota(monkeypatch, {"namespace": "a", "team": None, "quota": None})
    result = runner.invoke(cli.app, ["set", "a", "--pods", "12"])
    assert result.exit_code == 0, result.output
    assert captured["pods"] == 12


def test_set_wuji_error_friendly_exit(monkeypatch):
    _guard_no_kube(monkeypatch)

    def boom(*a, **k):
        raise cli.WujiError("需要管理员 kubeconfig")

    monkeypatch.setattr(cli.sdk, "set_quota", boom)
    result = runner.invoke(cli.app, ["set", "alice", "--gpu", "8"])
    assert result.exit_code == 1
    assert "需要管理员" in result.output
    assert not isinstance(result.exception, cli.WujiError)  # no leaked traceback


# =========================================================================== #
# volcano set-queue  (per-team Volcano Queue)
# =========================================================================== #
def test_set_queue_deserved_and_cap_threaded(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_set_queue(monkeypatch, {
        "name": "redteam", "created": True,
        "queue": {"deserved": "4", "capability": "8", "reclaimable": True, "state": "Open"},
    })
    result = runner.invoke(cli.app, ["set-queue", "redteam", "--deserved", "4", "--cap", "8"])
    assert result.exit_code == 0, result.output
    assert captured["team"] == "redteam"
    assert captured["deserved"] == 4 and captured["cap"] == 8
    assert "已创建队列 redteam" in result.output
    assert "deserved=4" in result.output and "cap=8" in result.output


def test_set_queue_short_flags(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_set_queue(monkeypatch, {
        "name": "t", "created": False,
        "queue": {"deserved": "2", "capability": "5", "reclaimable": True, "state": "Open"},
    })
    result = runner.invoke(cli.app, ["set-queue", "t", "-d", "2", "-c", "5"])
    assert result.exit_code == 0, result.output
    assert captured["deserved"] == 2 and captured["cap"] == 5
    assert "已更新队列 t" in result.output


def test_set_queue_delete(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_set_queue(monkeypatch, {"name": "redteam", "deleted": True})
    result = runner.invoke(cli.app, ["set-queue", "redteam", "--delete"])
    assert result.exit_code == 0, result.output
    assert captured["delete"] is True
    assert "已删除队列: redteam" in result.output


def test_set_queue_no_reclaimable_passes_false(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_set_queue(monkeypatch, {
        "name": "redteam", "created": False,
        "queue": {"deserved": "4", "capability": "8", "reclaimable": False, "state": "Open"},
    })
    result = runner.invoke(cli.app, ["set-queue", "redteam", "--no-reclaimable"])
    assert result.exit_code == 0, result.output
    assert captured["reclaimable"] is False


def test_set_queue_reclaimable_default_none(monkeypatch):
    # No --reclaimable/--no-reclaimable flag -> None passed (SDK keeps existing).
    _guard_no_kube(monkeypatch)
    captured = _capture_set_queue(monkeypatch, {
        "name": "redteam", "created": True,
        "queue": {"deserved": "4", "capability": "-", "reclaimable": True, "state": "Open"},
    })
    result = runner.invoke(cli.app, ["set-queue", "redteam", "--deserved", "4"])
    assert result.exit_code == 0, result.output
    assert captured["reclaimable"] is None


def test_set_queue_wuji_error_friendly_exit(monkeypatch):
    _guard_no_kube(monkeypatch)

    def boom(*a, **k):
        raise cli.WujiError("队列 default 删不掉")

    monkeypatch.setattr(cli.sdk, "set_queue", boom)
    result = runner.invoke(cli.app, ["set-queue", "default", "--delete"])
    assert result.exit_code == 1
    assert "删不掉" in result.output


def test_set_queue_view_existing_no_created_key(monkeypatch):
    # SDK view mode returns a dict WITHOUT "created" -> CLI prints现状, not 已更新.
    _guard_no_kube(monkeypatch)
    captured = _capture_set_queue(monkeypatch, {
        "name": "redteam",
        "queue": {"deserved": "4", "capability": "8", "reclaimable": True, "state": "Open"},
    })
    result = runner.invoke(cli.app, ["set-queue", "redteam"])
    assert result.exit_code == 0, result.output
    assert captured["deserved"] is None and captured["cap"] is None
    assert captured["reclaimable"] is None and captured["delete"] is False
    assert "队列 redteam: deserved=4" in result.output
    assert "已更新" not in result.output and "已创建" not in result.output


def test_set_queue_view_absent_prints_not_exist(monkeypatch):
    _guard_no_kube(monkeypatch)
    _capture_set_queue(monkeypatch, {"name": "ghost", "queue": None})
    result = runner.invoke(cli.app, ["set-queue", "ghost"])
    assert result.exit_code == 0, result.output
    assert "不存在" in result.output


# =========================================================================== #
# volcano whoami
# =========================================================================== #
def test_whoami_admin_line(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(cli.sdk, "whoami", lambda: {
        "namespace": "alice", "is_admin": True,
        "team": "redteam", "default_queue": "redteam",
    })
    result = runner.invoke(cli.app, ["whoami"])
    assert result.exit_code == 0, result.output
    assert "alice" in result.output
    assert "redteam" in result.output
    assert "管理员" in result.output


def test_whoami_non_admin_no_ns(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(cli.sdk, "whoami", lambda: {
        "namespace": None, "is_admin": False,
        "team": None, "default_queue": "default",
    })
    result = runner.invoke(cli.app, ["whoami"])
    assert result.exit_code == 0, result.output
    assert "无默认 namespace" in result.output
    assert "default" in result.output
    assert "普通用户" in result.output


def test_whoami_wuji_error_friendly_exit(monkeypatch):
    _guard_no_kube(monkeypatch)

    def boom():
        raise cli.WujiError("找不到 Kubernetes 配置")

    monkeypatch.setattr(cli.sdk, "whoami", boom)
    result = runner.invoke(cli.app, ["whoami"])
    assert result.exit_code == 1
    assert "找不到 Kubernetes 配置" in result.output


# =========================================================================== #
# default-queue routing (train / dev, no -q -> resolve_default_queue)
# =========================================================================== #
def _capture_submit(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli.sdk, "submit", lambda **kw: captured.update(kw) or {})
    return captured


def test_train_no_queue_uses_resolved_default(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_submit(monkeypatch)
    monkeypatch.setattr(cli.sdk, "resolve_default_queue", lambda team=None: "teamX")
    result = runner.invoke(
        cli.app, ["train", "-n", "j", "-t", "teamx", "-i", "img", "--cmd", "c"]
    )
    assert result.exit_code == 0, result.output
    assert captured["queue"] == "teamX"
    assert "queue=teamX" in result.output


def test_train_no_queue_threads_team_to_resolver(monkeypatch):
    _guard_no_kube(monkeypatch)
    _capture_submit(monkeypatch)
    seen = {}
    monkeypatch.setattr(
        cli.sdk, "resolve_default_queue",
        lambda team=None: seen.setdefault("team", team) or "teamX",
    )
    result = runner.invoke(
        cli.app, ["train", "-n", "j", "-t", "teamx", "-i", "img", "--cmd", "c"]
    )
    assert result.exit_code == 0, result.output
    assert seen["team"] == "teamx"


def test_train_explicit_queue_bypasses_resolver(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_submit(monkeypatch)
    monkeypatch.setattr(
        cli.sdk, "resolve_default_queue",
        lambda team=None: (_ for _ in ()).throw(AssertionError("resolver must not run with -q")),
    )
    result = runner.invoke(
        cli.app, ["train", "-n", "j", "-t", "teamx", "-i", "img", "--cmd", "c", "-q", "qqq"]
    )
    assert result.exit_code == 0, result.output
    assert captured["queue"] == "qqq"


def test_train_dry_run_renders_resolved_queue(monkeypatch):
    monkeypatch.setattr(cli.sdk, "resolve_default_queue", lambda team=None: "teamX")
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
    assert manifest["spec"]["queue"] == "teamX"


def test_train_dry_run_explicit_queue_bypasses_resolver(monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "resolve_default_queue",
        lambda team=None: (_ for _ in ()).throw(AssertionError("resolver must not run with -q")),
    )
    result = runner.invoke(
        cli.app,
        ["train", "-n", "dj", "-t", "teamx", "-i", "img", "--cmd", "c", "-q", "myq", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load(result.output)
    assert manifest["spec"]["queue"] == "myq"


def test_dev_no_queue_uses_resolved_default(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_submit(monkeypatch)
    monkeypatch.setattr(cli.sdk, "resolve_default_queue", lambda team=None: "devteam")
    result = runner.invoke(
        cli.app, ["dev", "-n", "d", "-t", "teamx", "-i", "img", "--no-expose-ssh"]
    )
    assert result.exit_code == 0, result.output
    assert captured["queue"] == "devteam"


def test_dev_dry_run_renders_resolved_queue(monkeypatch):
    monkeypatch.setattr(cli.sdk, "resolve_default_queue", lambda team=None: "devteam")
    monkeypatch.setattr(
        cli.sdk, "submit",
        lambda **kw: (_ for _ in ()).throw(AssertionError("no submit on dry-run")),
    )
    result = runner.invoke(
        cli.app, ["dev", "-n", "dj", "-t", "teamx", "-i", "img", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load(result.output)
    assert manifest["spec"]["queue"] == "devteam"


def test_dev_explicit_queue_bypasses_resolver(monkeypatch):
    _guard_no_kube(monkeypatch)
    captured = _capture_submit(monkeypatch)
    monkeypatch.setattr(
        cli.sdk, "resolve_default_queue",
        lambda team=None: (_ for _ in ()).throw(AssertionError("resolver must not run with -q")),
    )
    result = runner.invoke(
        cli.app,
        ["dev", "-n", "d", "-t", "teamx", "-i", "img", "--no-expose-ssh", "-q", "explicitq"],
    )
    assert result.exit_code == 0, result.output
    assert captured["queue"] == "explicitq"
