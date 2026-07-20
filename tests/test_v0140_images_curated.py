"""v0.14.0 — curated `volcano images` display + top-level `--version`.

Two source changes land in this release (see volcano/cli.py):

1. `volcano images` no longer prints a flat "── 基础镜像(base)──" list. It
   curates rows into three buckets:
     ⭐ 推荐基础镜像  = base rows where `_is_recommended_base(image)`
                        (platform public lib `.../wuji-rl/wuji-rl/*`, or
                         `ubuntu:` / `pytorch/pytorch:` from docker.io)
     ── 其它基础镜像 = the remaining base rows (team/historical work images that
                        acr-sync auto-collected as base)
     ── 团队保存    = rows whose kind != "base"
   Recommended rows sort platform-lib (`/wuji-rl/wuji-rl/`) before docker.io.

2. A root `--version` / `-V` eager callback prints `volcano <version>` and exits.

Everything is mocked (`cli.sdk.list_images`, `cli.current_namespace`); no cluster
or kubeconfig is touched. Tests live only under tests/.
"""

import pytest
from typer.testing import CliRunner

import volcano
import volcano.cli as cli

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_kube(monkeypatch):
    # Any accidental cluster access blows up loudly rather than reaching out.
    monkeypatch.setattr(cli, "load_clients", lambda: (_ for _ in ()).throw(
        AssertionError("load_clients must not be called in CLI unit tests")
    ))


def _row(image, kind="base", team="", node="", created=""):
    return {"image": image, "kind": kind, "team": team, "node": node, "created": created}


# --------------------------------------------------------------------------- #
# 1) _is_recommended_base — the classification predicate
# --------------------------------------------------------------------------- #
ACR = "wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com"


@pytest.mark.parametrize("image", [
    # ① platform public lib (double wuji-rl layer) — the images WE save
    f"{ACR}/wuji-rl/wuji-rl/ubuntu-sshd:v1",
    f"{ACR}/wuji-rl/wuji-rl/pytorch-sshd:v2",
    f"{ACR}/wuji-rl/wuji-rl/pytorch-devel-sshd:latest",
    # ② docker.io generic ubuntu
    "ubuntu:24.04",
    "ubuntu:22.04",
    # ③ docker.io generic pytorch
    "pytorch/pytorch:2.11.0-cuda12.1-cudnn9-runtime",
])
def test_is_recommended_base_true(image):
    assert cli._is_recommended_base(image) is True


@pytest.mark.parametrize("image", [
    # ④ team work images — single wuji-rl layer, NOT /wuji-rl/wuji-rl/
    f"{ACR}/wuji-rl/wuji-mjlab:v1",
    f"{ACR}/wuji-rl/ecrl:v1",
    f"{ACR}/wuji-rl/wuji-e2e/e2e-check:v1",
    # ⑤ arbitrary third-party
    "docker.io/library/redis:7",
    "nginx:1.27",
    "",
])
def test_is_recommended_base_false(image):
    assert cli._is_recommended_base(image) is False


def test_recommended_predicate_returns_bool_not_truthy():
    # Guard against a regression to a truthy-but-non-bool return.
    assert cli._is_recommended_base("ubuntu:22.04") is True
    assert cli._is_recommended_base("nginx:1") is False


@pytest.mark.parametrize("bad", [123, None, ["x"], {"image": "y"}, 3.14, object()])
def test_is_recommended_base_non_string_returns_false_no_crash(bad):
    # A hand-seeded index with a non-string image must not crash `volcano images`
    # (mirrors sdk's str() defence). Guard returns False rather than raising.
    assert cli._is_recommended_base(bad) is False


# --------------------------------------------------------------------------- #
# 2) images — three-bucket curation on a mixed index
# --------------------------------------------------------------------------- #
def _mixed_rows():
    return [
        # platform public lib (recommended, should top the ⭐ section)
        _row(f"{ACR}/wuji-rl/wuji-rl/pytorch-sshd:v2"),
        _row(f"{ACR}/wuji-rl/wuji-rl/ubuntu-sshd:v1"),
        # docker.io generics (recommended, but after platform-lib)
        _row("ubuntu:24.04"),
        _row("pytorch/pytorch:2.11.0-cuda12.1-cudnn9-runtime"),
        # team work images auto-collected as base (NOT recommended → 其它基础)
        _row(f"{ACR}/wuji-rl/wuji-mjlab:v1"),
        _row(f"{ACR}/wuji-rl/ecrl:v1"),
        # team-saved images (kind != base → 团队保存)
        _row(f"{ACR}/wuji-rl/myenv:v9", kind="saved", team="teamx", created="2027-01-02"),
        _row(f"{ACR}/wuji-rl/other:v3", kind="saved", team="teamx", created="2027-01-03"),
    ]


def test_images_three_buckets_and_counts(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: _mixed_rows())
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Summary line: 推荐 4 (2 platform-lib + ubuntu + pytorch), 其它基础 2, 团队保存 2
    assert "可拉取镜像:推荐 4、其它基础 2、团队保存 2" in out

    # Section headers all present.
    assert "⭐ 推荐基础镜像" in out
    assert "── 其它基础镜像" in out
    assert "── 团队保存(saved)──" in out

    # Usage footer unchanged.
    assert "volcano dev" in out and "volcano train" in out


def test_images_recommended_section_only_recommended_platform_lib_first(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: _mixed_rows())
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output

    rec_hdr = out.index("⭐ 推荐基础镜像")
    other_hdr = out.index("── 其它基础镜像")
    saved_hdr = out.index("── 团队保存(saved)──")
    rec_block = out[rec_hdr:other_hdr]

    # Recommended block lists exactly the recommended images…
    assert "wuji-rl/wuji-rl/pytorch-sshd:v2" in rec_block
    assert "wuji-rl/wuji-rl/ubuntu-sshd:v1" in rec_block
    assert "ubuntu:24.04" in rec_block
    assert "pytorch/pytorch:2.11.0" in rec_block
    # …and none of the team work / saved ones.
    assert "wuji-mjlab" not in rec_block
    assert "ecrl:v1" not in rec_block
    assert "myenv:v9" not in rec_block

    # platform-lib (/wuji-rl/wuji-rl/) sorts before docker.io generics.
    first_platform = min(
        rec_block.index("wuji-rl/wuji-rl/pytorch-sshd:v2"),
        rec_block.index("wuji-rl/wuji-rl/ubuntu-sshd:v1"),
    )
    last_platform = max(
        rec_block.index("wuji-rl/wuji-rl/pytorch-sshd:v2"),
        rec_block.index("wuji-rl/wuji-rl/ubuntu-sshd:v1"),
    )
    assert last_platform < rec_block.index("ubuntu:24.04")
    assert last_platform < rec_block.index("pytorch/pytorch:2.11.0")
    # within platform-lib, sorted by image string (pytorch-sshd < ubuntu-sshd)
    assert rec_block.index("wuji-rl/wuji-rl/pytorch-sshd:v2") == first_platform


def test_images_other_section_has_team_work_images(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: _mixed_rows())
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output

    other_hdr = out.index("── 其它基础镜像")
    saved_hdr = out.index("── 团队保存(saved)──")
    other_block = out[other_hdr:saved_hdr]

    assert "wuji-rl/wuji-mjlab:v1" in other_block
    assert "wuji-rl/ecrl:v1" in other_block
    # recommended / saved must not leak in
    assert "wuji-rl/wuji-rl/" not in other_block
    assert "myenv:v9" not in other_block


def test_images_saved_section_shows_team_and_created(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: _mixed_rows())
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output

    saved_hdr = out.index("── 团队保存(saved)──")
    saved_block = out[saved_hdr:]

    assert "wuji-rl/myenv:v9" in saved_block
    assert "wuji-rl/other:v3" in saved_block
    # team + created rendered in the trailing [ ... ] annotation
    assert "teamx" in saved_block
    assert "2027-01-02" in saved_block
    assert "2027-01-03" in saved_block


# --------------------------------------------------------------------------- #
# 3) images — boundary shapes
# --------------------------------------------------------------------------- #
def test_images_all_recommended_omits_other_section(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: [
        _row(f"{ACR}/wuji-rl/wuji-rl/ubuntu-sshd:v1"),
        _row("ubuntu:24.04"),
    ])
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "可拉取镜像:推荐 2、其它基础 0、团队保存 0" in out
    assert "⭐ 推荐基础镜像" in out
    assert "── 其它基础镜像" not in out   # empty bucket → header suppressed
    assert "── 团队保存(saved)──" not in out


def test_images_only_saved_no_base(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: [
        _row(f"{ACR}/wuji-rl/myenv:v9", kind="saved", team="teamx", created="2027"),
    ])
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "可拉取镜像:推荐 0、其它基础 0、团队保存 1" in out
    assert "⭐ 推荐基础镜像" not in out
    assert "── 其它基础镜像" not in out
    assert "── 团队保存(saved)──" in out
    assert "wuji-rl/myenv:v9" in out


def test_images_only_other_base_no_recommended(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: [
        _row(f"{ACR}/wuji-rl/wuji-mjlab:v1"),
    ])
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "可拉取镜像:推荐 0、其它基础 1、团队保存 0" in out
    assert "⭐ 推荐基础镜像" not in out
    assert "── 其它基础镜像" in out


def test_images_empty_index_unchanged_hint(monkeypatch):
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: [])
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert "镜像索引为空" in result.output
    # curated headers must NOT appear on the empty path
    assert "⭐ 推荐基础镜像" not in result.output


# --------------------------------------------------------------------------- #
# 4) images — --mine / --team threading (unchanged behaviour)
# --------------------------------------------------------------------------- #
def test_images_mine_passes_current_namespace(monkeypatch):
    seen = {}

    def fake_list(team=None):
        seen["team"] = team
        return []

    monkeypatch.setattr(cli.sdk, "list_images", fake_list)
    monkeypatch.setattr(cli, "current_namespace", lambda team=None: "wuji-rl")
    result = runner.invoke(cli.app, ["images", "--mine"])
    assert result.exit_code == 0, result.output
    assert seen["team"] == "wuji-rl"   # filter_team resolved from current ns


def test_images_team_flag_threaded_without_mine(monkeypatch):
    seen = {}

    def fake_list(team=None):
        seen["team"] = team
        return []

    monkeypatch.setattr(cli.sdk, "list_images", fake_list)
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen["team"] == "teamx"   # passed straight through


def test_images_error_exits_nonzero(monkeypatch):
    def boom(team=None):
        raise cli.WujiError("权限不足(403)")

    monkeypatch.setattr(cli.sdk, "list_images", boom)
    result = runner.invoke(cli.app, ["images", "-t", "teamx"])
    assert result.exit_code == 1
    assert "权限不足" in result.output


# --------------------------------------------------------------------------- #
# 5) top-level --version / -V
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_and_exits_zero(flag):
    result = runner.invoke(cli.app, [flag])
    assert result.exit_code == 0, result.output
    assert f"volcano {volcano.__version__}" in result.output


def test_version_matches_package_dunder():
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0, result.output
    assert f"volcano {volcano.__version__}" in result.output
    assert volcano.__version__ == "0.16.0"


def test_no_args_still_shows_help_after_root_callback():
    # Adding @app.callback() must not break no_args_is_help.
    result = runner.invoke(cli.app, [])
    # typer emits help then exits; help text lists the subcommands.
    assert "images" in result.output
    assert "train" in result.output or "submit" in result.output


@pytest.mark.parametrize("cmd", [["queue", "--help"], ["teams", "--help"]])
def test_subcommands_still_work_after_root_callback(cmd):
    # The root callback must not swallow subcommands.
    result = runner.invoke(cli.app, cmd)
    assert result.exit_code == 0, result.output


def test_version_flag_does_not_call_subcommand_sdk(monkeypatch):
    # --version is eager: it must short-circuit before any command body runs.
    monkeypatch.setattr(cli.sdk, "list_images", lambda team=None: (_ for _ in ()).throw(
        AssertionError("sdk must not be touched by --version")
    ))
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0, result.output
    assert f"volcano {volcano.__version__}" in result.output
