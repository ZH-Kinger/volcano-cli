"""CLI-layer tests for `wuji save` and `wuji images` (sdk mocked out).

Uses typer.testing.CliRunner. The sdk functions are patched, and load_clients is
guarded so nothing touches a real cluster. A team (`-t`) is always passed so
``current_namespace`` short-circuits without reading a kubeconfig.
"""

import pytest
from typer.testing import CliRunner

import wuji.cli as cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_kube(monkeypatch):
    monkeypatch.setattr(cli, "load_clients", lambda: (_ for _ in ()).throw(
        AssertionError("load_clients must not be called in CLI unit tests")
    ))


# --------------------------------------------------------------------------- #
# wuji save
# --------------------------------------------------------------------------- #
def test_save_no_wait_prints_request_name(monkeypatch):
    captured = {}

    def fake_save(name, **kw):
        captured["name"] = name
        captured.update(kw)
        return {"request": "wuji-save-j-w0-abc123", "image": "acr.io/wuji-rl/myenv:v1",
                "status": "Pending", "node": "", "message": ""}

    monkeypatch.setattr(cli.sdk, "save_image", fake_save)
    result = runner.invoke(
        cli.app,
        ["save", "j", "--tag", "myenv:v1", "-t", "teamx", "--no-wait"],
    )
    assert result.exit_code == 0, result.output
    assert captured["name"] == "j"
    assert captured["tag"] == "myenv:v1"
    assert captured["wait"] is False  # --no-wait
    assert "wuji-save-j-w0-abc123" in result.output   # request name shown
    assert "acr.io/wuji-rl/myenv:v1" in result.output  # target image shown


def test_save_success_prints_image_and_reuse_hint(monkeypatch):
    monkeypatch.setattr(cli.sdk, "save_image", lambda name, **kw: {
        "request": "req-1", "image": "acr.io/wuji-rl/myenv:v9",
        "status": "Succeeded", "node": "wuji-3", "message": "pushed",
    })
    result = runner.invoke(
        cli.app,
        ["save", "j", "--tag", "myenv:v9", "-t", "teamx"],
    )
    assert result.exit_code == 0, result.output
    assert "acr.io/wuji-rl/myenv:v9" in result.output   # saved image
    assert "wuji-3" in result.output                     # node
    assert "wuji dev" in result.output                   # reuse hint
    assert "wuji train" in result.output


def test_save_threads_worker_container_timeout(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli.sdk, "save_image",
        lambda name, **kw: captured.update(name=name, **kw) or {
            "request": "r", "image": "i:v1", "status": "Pending",
            "node": "", "message": "",
        },
    )
    result = runner.invoke(
        cli.app,
        ["save", "j", "--tag", "i:v1", "-t", "teamx", "--worker", "2",
         "--container", "main", "--timeout", "60", "--no-wait"],
    )
    assert result.exit_code == 0, result.output
    assert captured["worker"] == 2
    assert captured["container"] == "main"
    assert captured["timeout"] == 60
    assert captured["team"] == "teamx"


def test_save_error_exits_nonzero(monkeypatch):
    def boom(name, **kw):
        raise cli.WujiError("找不到容器 j-worker-0")

    monkeypatch.setattr(cli.sdk, "save_image", boom)
    result = runner.invoke(
        cli.app,
        ["save", "j", "--tag", "e:v1", "-t", "teamx"],
    )
    assert result.exit_code == 1
    assert "找不到容器" in result.output


def test_save_requires_tag(monkeypatch):
    # --tag is a required option; missing it must error before any sdk call
    monkeypatch.setattr(cli.sdk, "save_image",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no save")))
    result = runner.invoke(cli.app, ["save", "j", "-t", "teamx"])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# wuji images
# --------------------------------------------------------------------------- #
def test_images_empty_prints_hint(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: [])
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert "镜像索引为空" in result.output


def test_images_lists_rows(monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")  # keep rich from truncating cells
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: [
        {"image": "base-a:v1", "kind": "base", "team": "", "node": "", "created": "2026"},
        {"image": "mine-z:v1", "kind": "saved", "team": "teamx", "node": "n1", "created": "2027"},
    ])
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert "base-a:v1" in result.output
    assert "mine-z:v1" in result.output
    assert "wuji dev" in result.output  # usage footer


def test_images_team_flag_threaded(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.sdk, "list_images",
                        lambda team=None: (seen.update(team=team), [])[1])
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen["team"] == "teamx"  # without --mine, team passed straight through


def test_images_error_exits_nonzero(monkeypatch):
    def boom(team=None):
        raise cli.WujiError("权限不足(403)")

    monkeypatch.setattr(cli.sdk, "list_images", boom)
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 1
    assert "权限不足" in result.output
