"""Regression tests for v0.5.0 changes (relative to v0.4.0).

Covers the two documented deltas — all mocked, no real cluster / kubeconfig:

  1. `--mount <pvc>:<path>[:ro]` on train/dev:
     build_volcano_job(mounts=[...]) appends one volume + volumeMount per spec
     (name `mnt-i`, claimName=<pvc>, mountPath=<path>), maps `:ro` -> readOnly
     True / `:rw` or omitted -> False, and rejects malformed specs (wrong segment
     count, third segment not ro/rw, empty pvc, relative / root path, collision
     with the built-in mounts) with ValueError. mounts empty/None leaves the two
     original volumes (ws + dshm) untouched. CLI: `dev/train --mount ... --dry-run`
     renders both volumes; an illegal `--mount` exits non-zero with a friendly
     error (no traceback).
  2. `volcano volumes` (alias `vols`): sdk.list_pvcs(team=) reads PVCs via
     list_namespaced_persistent_volume_claim and returns
     {name,status,capacity,storageclass,volume}; the CLI prints a table with the
     header and PVC names.

Only tests/ is touched; source/config untouched.
"""

import types

import pytest
import yaml
from typer.testing import CliRunner

import volcano.cli as cli
import volcano.sdk as sdk
from volcano.job import build_volcano_job

runner = CliRunner()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _build(**overrides):
    """build_volcano_job with sane required defaults; override per test."""
    kw = dict(
        name="j", team="teamx", queue="default", image="img",
        command="python train.py",
    )
    kw.update(overrides)
    return build_volcano_job(**kw)


def _vols_by_name(manifest):
    return {v["name"]: v for v in manifest["spec"]["tasks"][0]["template"]["spec"]["volumes"]}


def _mounts_by_name(manifest):
    container = manifest["spec"]["tasks"][0]["template"]["spec"]["containers"][0]
    return {m["name"]: m for m in container["volumeMounts"]}


def _guard_no_kube(monkeypatch):
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )


@pytest.fixture(autouse=True)
def _stub_resolve_default_queue(monkeypatch):
    # resolve_default_queue is fail-closed now (raises without a real
    # team-labelled namespace + matching Queue — see
    # test_v090_sdk_quota_queue.py / test_v090_cli_quota_queue.py for that).
    # These tests pass -t teamx, a namespace that doesn't exist for real, and
    # don't care about queue resolution itself, so stub it rather than hit
    # the live cluster.
    monkeypatch.setattr(cli.sdk, "resolve_default_queue", lambda team=None: "default")


# --------------------------------------------------------------------------- #
# 1a. build_volcano_job(mounts=...) — happy paths
# --------------------------------------------------------------------------- #
def test_single_mount_ro_adds_volume_and_mount_readonly_true():
    m = _build(mounts=["public:/data/pub:ro"])
    vols = _vols_by_name(m)
    mounts = _mounts_by_name(m)
    # volume
    assert vols["mnt-0"]["persistentVolumeClaim"]["claimName"] == "public"
    assert vols["mnt-0"]["persistentVolumeClaim"]["readOnly"] is True
    # volumeMount
    assert mounts["mnt-0"]["mountPath"] == "/data/pub"
    assert mounts["mnt-0"]["readOnly"] is True


def test_single_mount_rw_is_readonly_false():
    m = _build(mounts=["public:/data/pub:rw"])
    assert _vols_by_name(m)["mnt-0"]["persistentVolumeClaim"]["readOnly"] is False
    assert _mounts_by_name(m)["mnt-0"]["readOnly"] is False


def test_single_mount_omitted_third_segment_defaults_rw():
    m = _build(mounts=["public:/data/pub"])
    assert _vols_by_name(m)["mnt-0"]["persistentVolumeClaim"]["readOnly"] is False
    assert _mounts_by_name(m)["mnt-0"]["readOnly"] is False


def test_multiple_mounts_indexed_and_correct():
    m = _build(mounts=["public:/a", "oss:/b:ro", "cache:/c:rw"])
    vols = _vols_by_name(m)
    mounts = _mounts_by_name(m)
    # three extra volumes, correctly numbered
    assert vols["mnt-0"]["persistentVolumeClaim"]["claimName"] == "public"
    assert vols["mnt-1"]["persistentVolumeClaim"]["claimName"] == "oss"
    assert vols["mnt-2"]["persistentVolumeClaim"]["claimName"] == "cache"
    assert mounts["mnt-0"]["mountPath"] == "/a" and mounts["mnt-0"]["readOnly"] is False
    assert mounts["mnt-1"]["mountPath"] == "/b" and mounts["mnt-1"]["readOnly"] is True
    assert mounts["mnt-2"]["mountPath"] == "/c" and mounts["mnt-2"]["readOnly"] is False


def test_mount_case_insensitive_ro():
    # third segment is lower-cased before comparison
    m = _build(mounts=["public:/x:RO"])
    assert _mounts_by_name(m)["mnt-0"]["readOnly"] is True


def test_mounts_coexist_with_shared_data_datasets_volume():
    # datasets volume is inserted when shared_data is set; mounts still index from 0
    m = _build(shared_data="corpus", mounts=["public:/x:ro"])
    vols = _vols_by_name(m)
    assert "datasets" in vols  # shared read-only dataset PVC present
    assert vols["mnt-0"]["persistentVolumeClaim"]["claimName"] == "public"
    assert _mounts_by_name(m)["mnt-0"]["mountPath"] == "/x"


# --------------------------------------------------------------------------- #
# 1b. build_volcano_job(mounts=...) — rejection paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    "onlyonesegment",          # 1 segment (no path)
    "pvc:/p:ro:extra",         # 4 segments
    "pvc:/p:xx",               # third segment not ro/rw
    ":/p",                     # empty pvc
    "  :/p",                   # whitespace-only pvc
    "pvc:relative/path",       # relative path
    "pvc:/",                   # root path
])
def test_illegal_mount_specs_raise_valueerror(bad):
    with pytest.raises(ValueError):
        _build(mounts=[bad])


@pytest.mark.parametrize("reserved", ["/workspace", "/datasets", "/dev/shm"])
def test_mount_collision_with_builtin_paths_raises(reserved):
    with pytest.raises(ValueError):
        _build(mounts=[f"public:{reserved}"])


def test_mount_collision_with_custom_workspace_path_raises():
    # workspace_path is part of the reserved set, so a custom NAS path collides too
    with pytest.raises(ValueError):
        _build(workspace_path="/mnt/nas", mounts=["public:/mnt/nas"])


def test_illegal_mount_raises_before_any_partial_state():
    # a valid mount followed by an invalid one must still raise (whole call fails)
    with pytest.raises(ValueError):
        _build(mounts=["good:/ok", "bad-no-path"])


# --------------------------------------------------------------------------- #
# 1b-hardening. normalization / `..` / duplicate-path rejection
# (auditor LOW#1/#2 follow-up: posixpath.normpath before compare + dedup)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bypass", [
    "public:/workspace/",     # trailing slash -> normpath -> /workspace (reserved)
    "public:/dev/shm/./",     # ./ + trailing slash -> /dev/shm (reserved)
    "public:/datasets/.",     # trailing /. -> /datasets (reserved)
])
def test_mount_reserved_bypass_after_normalization_raises(bypass):
    with pytest.raises(ValueError):
        _build(mounts=[bypass])


@pytest.mark.parametrize("dotdot", [
    "public:/a/../b",         # interior .. segment
    "public:/..",             # leading .. escaping root
    "public:/x/y/../../z",    # multiple .. segments
])
def test_mount_dotdot_segment_raises(dotdot):
    with pytest.raises(ValueError):
        _build(mounts=[dotdot])


@pytest.mark.parametrize("dup", [
    ["a:/x", "b:/x"],                 # exact same path, different pvc
    ["a:/x", "b:/x/"],               # same path after normalizing trailing slash
    ["a:/opt/data", "b:/opt/./data"],  # same path after normalizing ./
    ["a:/x:ro", "b:/x:rw"],          # same path, differing ro/rw still a dup
])
def test_mount_duplicate_normalized_path_raises(dup):
    with pytest.raises(ValueError):
        _build(mounts=dup)


def test_mount_path_is_normalized_no_trailing_slash():
    m = _build(mounts=["public:/opt/data/"])
    assert _mounts_by_name(m)["mnt-0"]["mountPath"] == "/opt/data"


def test_mount_path_normalizes_dot_segments():
    m = _build(mounts=["public:/opt/./data//sub/"])
    assert _mounts_by_name(m)["mnt-0"]["mountPath"] == "/opt/data/sub"


def test_distinct_paths_after_normalization_are_allowed():
    # /x and /x/y are different targets -> both accepted, indexed mnt-0/mnt-1
    m = _build(mounts=["a:/x", "b:/x/y"])
    mounts = _mounts_by_name(m)
    assert mounts["mnt-0"]["mountPath"] == "/x"
    assert mounts["mnt-1"]["mountPath"] == "/x/y"


# --------------------------------------------------------------------------- #
# 1c. regression: mounts None / empty leaves the base 2 volumes untouched
# --------------------------------------------------------------------------- #
def _base_volume_names(m):
    return [v["name"] for v in m["spec"]["tasks"][0]["template"]["spec"]["volumes"]]


def test_mounts_none_keeps_only_ws_and_dshm():
    m = _build()  # mounts defaults to None
    assert _base_volume_names(m) == ["ws", "dshm"]
    mounts = _mounts_by_name(m)
    assert set(mounts) == {"ws", "dshm"}
    # no mnt-* leaked in
    assert not any(v.startswith("mnt-") for v in _vols_by_name(m))


def test_mounts_empty_list_keeps_only_ws_and_dshm():
    m = _build(mounts=[])
    assert _base_volume_names(m) == ["ws", "dshm"]
    assert not any(v.startswith("mnt-") for v in _vols_by_name(m))


def test_ws_volume_still_points_at_team_nas():
    # ensure the mounts feature did not disturb the standard NAS wiring
    m = _build(team="alpha", mounts=["public:/x"])
    assert _vols_by_name(m)["ws"]["persistentVolumeClaim"]["claimName"] == "alpha-nas"


# --------------------------------------------------------------------------- #
# 1d. CLI --mount dry-run rendering
# --------------------------------------------------------------------------- #
def test_dev_two_mounts_render_both_volumes(monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "submit",
        lambda **kw: (_ for _ in ()).throw(AssertionError("no submit on dry-run")),
    )
    result = runner.invoke(cli.app, [
        "dev", "-n", "d", "-t", "teamx", "-i", "img",
        "--mount", "a:/x", "--mount", "b:/y:ro", "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load(result.output)
    vols = _vols_by_name(manifest)
    mounts = _mounts_by_name(manifest)
    assert vols["mnt-0"]["persistentVolumeClaim"]["claimName"] == "a"
    assert vols["mnt-1"]["persistentVolumeClaim"]["claimName"] == "b"
    assert mounts["mnt-0"]["mountPath"] == "/x" and mounts["mnt-0"]["readOnly"] is False
    assert mounts["mnt-1"]["mountPath"] == "/y" and mounts["mnt-1"]["readOnly"] is True


def test_train_two_mounts_render_both_volumes(monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "submit",
        lambda **kw: (_ for _ in ()).throw(AssertionError("no submit on dry-run")),
    )
    result = runner.invoke(cli.app, [
        "train", "-n", "j", "-t", "teamx", "-i", "img", "--cmd", "python t.py",
        "--mount", "a:/x", "--mount", "b:/y:ro", "--dry-run",
    ])
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load(result.output)
    vols = _vols_by_name(manifest)
    assert vols["mnt-0"]["persistentVolumeClaim"]["claimName"] == "a"
    assert vols["mnt-1"]["persistentVolumeClaim"]["claimName"] == "b"
    assert vols["mnt-1"]["persistentVolumeClaim"]["readOnly"] is True


def test_cli_illegal_mount_dry_run_friendly_error(monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "submit",
        lambda **kw: (_ for _ in ()).throw(AssertionError("no submit on dry-run")),
    )
    result = runner.invoke(cli.app, [
        "dev", "-n", "d", "-t", "teamx", "-i", "img",
        "--mount", "no-path-here", "--dry-run",
    ])
    assert result.exit_code != 0
    # friendly one-liner, not a Python traceback
    assert "错误" in result.output
    assert "Traceback" not in result.output


def test_cli_illegal_mount_third_segment_dry_run_friendly_error(monkeypatch):
    monkeypatch.setattr(
        cli.sdk, "submit",
        lambda **kw: (_ for _ in ()).throw(AssertionError("no submit on dry-run")),
    )
    result = runner.invoke(cli.app, [
        "train", "-n", "j", "-t", "teamx", "-i", "img", "--cmd", "c",
        "--mount", "pvc:/x:banana", "--dry-run",
    ])
    assert result.exit_code != 0
    assert "错误" in result.output
    assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# 2a. sdk.list_pvcs
# --------------------------------------------------------------------------- #
def _pvc(name, phase="Bound", storage="100Gi", sc="nas", vol="pv-1"):
    """A fake kubernetes V1PersistentVolumeClaim-ish object (attribute access)."""
    status = types.SimpleNamespace(
        phase=phase,
        capacity={"storage": storage} if storage is not None else None,
    )
    spec = types.SimpleNamespace(storage_class_name=sc, volume_name=vol)
    metadata = types.SimpleNamespace(name=name)
    return types.SimpleNamespace(metadata=metadata, status=status, spec=spec)


@pytest.fixture
def fake_core(monkeypatch):
    box = {"items": []}
    seen = {}

    class Core:
        def list_namespaced_persistent_volume_claim(self, namespace):
            seen["namespace"] = namespace
            return types.SimpleNamespace(items=box["items"])

    monkeypatch.setattr(sdk, "load_clients", lambda: (Core(), None, None))
    monkeypatch.setattr(sdk, "current_namespace", lambda team=None: team or "teamx")
    return box, seen


def test_list_pvcs_structure(fake_core):
    box, seen = fake_core
    box["items"] = [
        _pvc("teamx-nas", storage="500Gi", sc="nfs", vol="pv-nas"),
        _pvc("public", storage="1Ti", sc="local-storage", vol="pv-pub"),
    ]
    out = sdk.list_pvcs(team="teamx")
    assert seen["namespace"] == "teamx"
    assert out == [
        {"name": "teamx-nas", "status": "Bound", "capacity": "500Gi",
         "storageclass": "nfs", "volume": "pv-nas"},
        {"name": "public", "status": "Bound", "capacity": "1Ti",
         "storageclass": "local-storage", "volume": "pv-pub"},
    ]


def test_list_pvcs_handles_missing_fields(fake_core):
    box, _ = fake_core
    # no capacity, no storageclass, no bound volume, pending
    p = _pvc("pending-pvc", phase="Pending", storage=None, sc=None, vol=None)
    box["items"] = [p]
    out = sdk.list_pvcs(team="teamx")
    assert out[0]["status"] == "Pending"
    assert out[0]["capacity"] == "-"
    assert out[0]["storageclass"] == "-"
    assert out[0]["volume"] == "-"


def test_list_pvcs_empty(fake_core):
    box, _ = fake_core
    box["items"] = []
    assert sdk.list_pvcs(team="teamx") == []


# --------------------------------------------------------------------------- #
# 2b. CLI `volumes` / `vols`
# --------------------------------------------------------------------------- #
def test_volumes_command_prints_table(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(cli, "current_namespace", lambda team=None: team or "teamx")
    monkeypatch.setattr(cli.sdk, "list_pvcs", lambda team=None: [
        {"name": "teamx-nas", "status": "Bound", "capacity": "500Gi",
         "storageclass": "nfs", "volume": "pv-nas"},
        {"name": "publicpvc", "status": "Bound", "capacity": "1Ti",
         "storageclass": "local", "volume": "pv-pub"},
    ])
    result = runner.invoke(cli.app, ["volumes", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert "NAME" in result.output          # table header
    assert "teamx-nas" in result.output     # a PVC name
    assert "publicpvc" in result.output


def test_vols_alias_works(monkeypatch):
    _guard_no_kube(monkeypatch)
    seen = {}
    monkeypatch.setattr(cli, "current_namespace", lambda team=None: team or "teamx")

    def fake_list(team=None):
        seen["team"] = team
        return [{"name": "onlypvc", "status": "Bound", "capacity": "10Gi",
                 "storageclass": "nfs", "volume": "pv-x"}]

    monkeypatch.setattr(cli.sdk, "list_pvcs", fake_list)
    result = runner.invoke(cli.app, ["vols", "-t", "teamx"])
    assert result.exit_code == 0, result.output
    assert seen["team"] == "teamx"
    assert "onlypvc" in result.output


def test_volumes_command_forwards_error_friendly(monkeypatch):
    _guard_no_kube(monkeypatch)
    from volcano.kube import WujiError

    monkeypatch.setattr(
        cli.sdk, "list_pvcs",
        lambda team=None: (_ for _ in ()).throw(WujiError("无法访问命名空间 teamx")),
    )
    result = runner.invoke(cli.app, ["volumes", "-t", "teamx"])
    assert result.exit_code != 0
    assert "错误" in result.output
    assert "Traceback" not in result.output


# --------------------------------------------------------------------------- #
# 1c. --mount 加固(采纳 auditor LOW#1/#2):归一化保留路径绕过 / '..' / 去重
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["public:/workspace/", "public:/dev/shm/./", "public:/datasets/."])
def test_mount_normalized_reserved_path_rejected(bad):
    """尾斜杠 / '.'段 归一化后命中内置挂载点 → 拒绝(不能绕过保留路径检查)。"""
    with pytest.raises(ValueError):
        _build(mounts=[bad])


@pytest.mark.parametrize("bad", ["public:/a/../b", "public:/..", "public:/x/../../etc"])
def test_mount_dotdot_segment_rejected(bad):
    """容器路径含 '..' 段一律拒绝。"""
    with pytest.raises(ValueError):
        _build(mounts=[bad])


@pytest.mark.parametrize("dup", [["a:/x", "b:/x"], ["a:/x", "b:/x/"], ["a:/x", "b:/x/."]])
def test_mount_duplicate_path_rejected(dup):
    """两个 --mount 归一化后落到同一路径 → 提前友好报错(而非 apply 时 k8s 拒)。"""
    with pytest.raises(ValueError):
        _build(mounts=dup)


def test_mount_path_is_normalized_no_trailing_slash():
    """归一化生效:/opt/data/ → mountPath 渲染为 /opt/data(无尾斜杠)。"""
    m = _build(mounts=["public:/opt/data/"])
    paths = [x["mountPath"] for x in _mounts_by_name(m).values()]
    assert "/opt/data" in paths and "/opt/data/" not in paths


def test_mount_custom_workspace_path_normalized_collision_rejected():
    """自定义 workspace_path 也参与归一化比对:尾斜杠变体仍算冲突。"""
    with pytest.raises(ValueError):
        _build(workspace_path="/ws", mounts=["public:/ws/"])
