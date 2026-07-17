"""v0.12.0: `volcano save` default flips from foreground-wait to background-submit.

Source change (volcano/cli.py `save`, NOT touched here):
    * `--wait` (bool, default False) replaces the old `--no-wait` toggle: default
      is now "submit in the background and return immediately".
    * `--no-wait` is kept as a hidden, no-effect compatibility flag (still yields
      the background branch, same as the default).
    * `--timeout` only matters when `--wait` is given.
    * Default / background branch calls `sdk.save_image(..., wait=False)` and
      prints "已提交后台保存 ..." + request name + target image +
      "volcano images --mine[ -t team]" + a `kubectl get cm` status hint +
      "volcano dev ..." reuse hint, then returns.
    * `--wait` branch calls with `wait=True` and prints "✅ 已保存并推送 ..."
      plus the dev/train reuse hints.

Everything runs through Typer's CliRunner with `sdk.save_image` mocked and
`load_clients` guarded (via a `-t team` on every invoke, so `current_namespace`
short-circuits without a kubeconfig). No cluster, no network, no real sleep.
Only tests/ is touched.
"""

import pytest
from typer.testing import CliRunner

import volcano.cli as cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_kube(monkeypatch):
    # Same guard as test_cli_save.py: any accidental cluster touch blows up loud.
    monkeypatch.setattr(cli, "load_clients", lambda: (_ for _ in ()).throw(
        AssertionError("load_clients must not be called in CLI unit tests")
    ))


def _mock_save(monkeypatch, *, node="wuji-3", status="Pending"):
    """Patch sdk.save_image to record kwargs and return a canned result."""
    captured = {}

    def fake_save(name, **kw):
        captured["name"] = name
        captured.update(kw)
        return {
            "request": "wuji-save-j-w0-deadbeef",
            "image": "acr.io/wuji-rl/myenv:v1",
            "status": status,
            "node": node,
            "message": "",
        }

    monkeypatch.setattr(cli.sdk, "save_image", fake_save)
    return captured


# --------------------------------------------------------------------------- #
# default (no flag) → background submit
# --------------------------------------------------------------------------- #
def test_default_passes_wait_false_to_sdk(monkeypatch):
    captured = _mock_save(monkeypatch)
    result = runner.invoke(cli.app, ["save", "j", "--tag", "myenv:v1", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert captured["wait"] is False


def test_default_prints_background_submission(monkeypatch):
    _mock_save(monkeypatch)
    result = runner.invoke(cli.app, ["save", "j", "--tag", "myenv:v1", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "已提交后台保存" in out                        # background banner
    assert "wuji-save-j-w0-deadbeef" in out              # request name
    assert "acr.io/wuji-rl/myenv:v1" in out              # target image
    assert "volcano images --mine" in out                # progress hint
    assert "volcano dev" in out                           # reuse hint
    # the foreground success line must NOT leak into the background path
    assert "✅ 已保存并推送" not in out


# --------------------------------------------------------------------------- #
# --wait → foreground
# --------------------------------------------------------------------------- #
def test_wait_passes_wait_true_to_sdk(monkeypatch):
    captured = _mock_save(monkeypatch, status="Succeeded")
    result = runner.invoke(
        cli.app, ["save", "j", "--tag", "myenv:v1", "-t", "teamx", "--wait"]
    )
    assert result.exit_code == 0, result.output
    assert captured["wait"] is True


def test_wait_prints_saved_and_reuse_hints(monkeypatch):
    _mock_save(monkeypatch, status="Succeeded")
    result = runner.invoke(
        cli.app, ["save", "j", "--tag", "myenv:v1", "-t", "teamx", "--wait"]
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "✅ 已保存并推送" in out                        # foreground success
    assert "acr.io/wuji-rl/myenv:v1" in out              # image
    assert "wuji-3" in out                                # node
    assert "volcano dev" in out                           # reuse hint
    assert "volcano train" in out                         # reuse hint
    # foreground must NOT emit the background banner
    assert "已提交后台保存" not in out


def test_wait_threads_timeout_to_sdk(monkeypatch):
    captured = _mock_save(monkeypatch, status="Succeeded")
    result = runner.invoke(
        cli.app,
        ["save", "j", "--tag", "myenv:v1", "-t", "teamx", "--wait", "--timeout", "42"],
    )
    assert result.exit_code == 0, result.output
    assert captured["wait"] is True
    assert captured["timeout"] == 42


# --------------------------------------------------------------------------- #
# --no-wait (hidden compat) → still background, same as default
# --------------------------------------------------------------------------- #
def test_no_wait_still_background_and_wait_false(monkeypatch):
    captured = _mock_save(monkeypatch)
    result = runner.invoke(
        cli.app, ["save", "j", "--tag", "myenv:v1", "-t", "teamx", "--no-wait"]
    )
    assert result.exit_code == 0, result.output
    assert captured["wait"] is False                     # compat flag = background
    assert "已提交后台保存" in result.output               # background branch
    assert "✅ 已保存并推送" not in result.output


def test_no_wait_matches_default_output(monkeypatch):
    # --no-wait must behave exactly like the (new) default.
    _mock_save(monkeypatch)
    default = runner.invoke(cli.app, ["save", "j", "--tag", "myenv:v1", "-t", "teamx"])
    _mock_save(monkeypatch)
    nowait = runner.invoke(
        cli.app, ["save", "j", "--tag", "myenv:v1", "-t", "teamx", "--no-wait"]
    )
    assert default.exit_code == 0 and nowait.exit_code == 0
    assert default.output == nowait.output


# --------------------------------------------------------------------------- #
# -t team threading into the background hints
# --------------------------------------------------------------------------- #
def test_team_arg_in_background_hints(monkeypatch):
    _mock_save(monkeypatch)
    result = runner.invoke(cli.app, ["save", "j", "--tag", "myenv:v1", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "volcano images --mine -t teamx" in out
    assert "-t teamx" in out
    # dev reuse hint should carry the team too
    assert "volcano dev -n <新开发机> -t teamx" in out


def test_no_team_background_hints_omit_team_arg(monkeypatch):
    # When no team is given, hints must not sprout a stray "-t" fragment.
    # current_namespace would otherwise read a kubeconfig, so stub it out.
    _mock_save(monkeypatch)
    monkeypatch.setattr(cli, "current_namespace", lambda team=None: "resolved-ns")
    result = runner.invoke(cli.app, ["save", "j", "--tag", "myenv:v1"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "已提交后台保存" in out
    assert "volcano images --mine" in out
    assert " -t " not in out                              # no dangling team flag
    assert "resolved-ns" in out                           # ns used in kubectl hint


# --------------------------------------------------------------------------- #
# error path → friendly non-zero exit (both branches)
# --------------------------------------------------------------------------- #
def test_default_sdk_error_dies_friendly(monkeypatch):
    def boom(name, **kw):
        raise cli.WujiError("找不到容器 j-worker-0")

    monkeypatch.setattr(cli.sdk, "save_image", boom)
    result = runner.invoke(cli.app, ["save", "j", "--tag", "e:v1", "-t", "teamx"])
    assert result.exit_code == 1
    assert "找不到容器" in result.output
    # no traceback leaked
    assert "Traceback" not in result.output


def test_wait_sdk_error_dies_friendly(monkeypatch):
    def boom(name, **kw):
        raise cli.WujiError("saver 超时(1800s)")

    monkeypatch.setattr(cli.sdk, "save_image", boom)
    result = runner.invoke(
        cli.app, ["save", "j", "--tag", "e:v1", "-t", "teamx", "--wait"]
    )
    assert result.exit_code == 1
    assert "saver 超时" in result.output


def test_no_wait_hidden_in_help(monkeypatch):
    # Compat flag is retained but must stay hidden; --wait must be documented.
    result = runner.invoke(cli.app, ["save", "--help"])
    assert result.exit_code == 0, result.output
    assert "--wait" in result.output
    assert "--no-wait" not in result.output
