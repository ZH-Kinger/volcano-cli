"""Unit tests for the wuji `save` surface in wuji.sdk.

Covers ``_resolve_image_ref`` (tag → full ACR ref rules), ``save_image``
(SaveRequest ConfigMap creation + status polling / error / timeout paths) and
``list_images`` (image-index ConfigMap parsing, team filter, sorting).

The kubernetes core client, ``time.sleep`` and ``time.time`` are all mocked —
no cluster and no real waiting. Only this tests/ tree is touched.
"""

import json
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

import wuji.sdk as sdk
from wuji.kube import (
    DEFAULT_ACR_REGISTRY,
    DEFAULT_ACR_REPO_PREFIX,
    SAVE_REQUEST_LABEL,
)

ACR = DEFAULT_ACR_REGISTRY
PREFIX = DEFAULT_ACR_REPO_PREFIX


# --------------------------------------------------------------------------- #
# _resolve_image_ref — team-scoped, mandatory team, short-name only
# --------------------------------------------------------------------------- #
def test_resolve_bare_name_with_tag_is_team_scoped():
    # the previously-buggy case: short 'name:tag' now correctly team-prefixed
    assert (
        sdk._resolve_image_ref("myenv:v1", team="teamA")
        == f"{ACR}/{PREFIX}/teamA/myenv:v1"
    )


def test_resolve_bare_repo_no_tag_gets_latest():
    assert (
        sdk._resolve_image_ref("myenv", team="teamA")
        == f"{ACR}/{PREFIX}/teamA/myenv:latest"
    )


def test_resolve_multisegment_relative_is_team_scoped():
    # first segment 'sub' has no '.'/':'/localhost → not a host → allowed, prefixed
    assert (
        sdk._resolve_image_ref("sub/myenv:v1", team="teamA")
        == f"{ACR}/{PREFIX}/teamA/sub/myenv:v1"
    )


def test_resolve_full_registry_ref_rejected():
    # tag carrying a registry host (has '/' and first segment has '.') → rejected
    with pytest.raises(sdk.WujiError) as ei:
        sdk._resolve_image_ref("registry.example.com/foo/bar:v2", team="teamA")
    assert "registry" in str(ei.value)


def test_resolve_default_acr_ref_rejected():
    # even the platform's own ACR host is rejected — only short names allowed
    with pytest.raises(sdk.WujiError):
        sdk._resolve_image_ref(f"{ACR}/{PREFIX}/x:v1", team="teamA")


def test_resolve_host_with_port_rejected():
    # first segment 'localhost:5000' looks like a host (':' / localhost) → rejected
    with pytest.raises(sdk.WujiError):
        sdk._resolve_image_ref("localhost:5000/img", team="teamA")


def test_resolve_empty_tag_raises():
    with pytest.raises(sdk.WujiError):
        sdk._resolve_image_ref("", team="teamA")


def test_resolve_whitespace_only_tag_raises():
    with pytest.raises(sdk.WujiError):
        sdk._resolve_image_ref("   ", team="teamA")


def test_resolve_empty_team_raises():
    with pytest.raises(sdk.WujiError):
        sdk._resolve_image_ref("myenv:v1", team="")


def test_resolve_strips_surrounding_slash_and_whitespace():
    # .strip().strip('/') on both ends → 'myenv:v1'
    assert (
        sdk._resolve_image_ref("  /myenv:v1/  ", team="teamA")
        == f"{ACR}/{PREFIX}/teamA/myenv:v1"
    )


def test_resolve_custom_registry_and_prefix():
    assert (
        sdk._resolve_image_ref("x:v3", team="teamA", registry="r.io", repo_prefix="p")
        == "r.io/p/teamA/x:v3"
    )


# --------------------------------------------------------------------------- #
# save_image
# --------------------------------------------------------------------------- #
def _pod(phase="Running"):
    return SimpleNamespace(status=SimpleNamespace(phase=phase))


class _Clock:
    """Monotonic fake time.time(): returns start, start+step, start+2*step, ..."""

    def __init__(self, start=0.0, step=5.0):
        self.t = start
        self.step = step

    def __call__(self):
        cur = self.t
        self.t += self.step
        return cur


class SaveCore:
    def __init__(self, *, pod=None, pod_exc=None, create_exc=None, cm_reads=None,
                 default_read=None):
        self._pod = pod if pod is not None else _pod("Running")
        self.pod_exc = pod_exc
        self.create_exc = create_exc
        self.created = []
        self._cm_reads = list(cm_reads or [])
        # what read returns once the scripted queue is exhausted (timeout path)
        self._default_read = default_read if default_read is not None else {"status": "Pending"}
        self.read_calls = 0

    def read_namespaced_pod(self, *, name, namespace):
        if self.pod_exc:
            raise self.pod_exc
        return self._pod

    def create_namespaced_config_map(self, *, namespace, body):
        if self.create_exc:
            raise self.create_exc
        self.created.append((namespace, body))
        return body

    def read_namespaced_config_map(self, *, name, namespace):
        self.read_calls += 1
        data = self._cm_reads.pop(0) if self._cm_reads else self._default_read
        return SimpleNamespace(data=dict(data))


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(sdk.time, "sleep", lambda *_a, **_k: None)


def _patch_core(monkeypatch, core):
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))


def test_save_happy_path_creates_request_configmap(monkeypatch, no_sleep):
    core = SaveCore(cm_reads=[
        {"status": "Pending"},
        {"status": "Succeeded", "node": "wuji-3", "message": "pushed"},
    ])
    _patch_core(monkeypatch, core)
    res = sdk.save_image("myjob", tag="myenv:v9", team="teamx")

    # one SaveRequest ConfigMap created in the team namespace
    assert len(core.created) == 1
    ns, body = core.created[0]
    assert ns == "teamx"
    md = body["metadata"]
    assert md["namespace"] == "teamx"
    assert md["labels"][SAVE_REQUEST_LABEL] == "true"
    assert md["labels"]["app.kubernetes.io/managed-by"] == "wuji"
    assert md["labels"]["wuji.io/job"] == "myjob"
    data = body["data"]
    assert data["pod"] == "myjob-worker-0"
    assert data["namespace"] == "teamx"
    assert data["container"] == "worker"
    # NEW: request carries the short tag (saver rebuilds authoritative ref) plus
    # the client-side predicted full, team-scoped image ref.
    assert data["tag"] == "myenv:v9"
    assert data["image"] == f"{ACR}/{PREFIX}/teamx/myenv:v9"
    assert data["image"] == res["image"]
    assert data["status"] == "Pending"
    # request name is echoed into the ConfigMap name
    assert md["name"] == res["request"]

    # polled Pending → Succeeded and returned the final outcome
    assert res["status"] == "Succeeded"
    assert res["node"] == "wuji-3"
    assert res["message"] == "pushed"
    # image is the team-scoped ACR ref, not the bare tag
    assert res["image"] == f"{ACR}/{PREFIX}/teamx/myenv:v9"


def test_save_request_data_has_short_tag_and_scoped_image(monkeypatch, no_sleep):
    core = SaveCore(cm_reads=[{"status": "Succeeded", "node": "n1"}])
    _patch_core(monkeypatch, core)
    res = sdk.save_image("j", tag="myenv:v1", team="teamB")
    _, body = core.created[0]
    assert body["data"]["tag"] == "myenv:v1"  # short name for the saver
    assert body["data"]["image"] == f"{ACR}/{PREFIX}/teamB/myenv:v1"  # full ref
    assert res["image"] == f"{ACR}/{PREFIX}/teamB/myenv:v1"


def test_save_rejects_full_registry_tag(monkeypatch, no_sleep):
    # a --tag carrying a registry host is refused before any pod/CM work
    core = SaveCore()
    _patch_core(monkeypatch, core)
    with pytest.raises(sdk.WujiError):
        sdk.save_image("j", tag="reg.com/x:v1", team="teamx")
    assert core.created == []  # nothing created


def test_save_worker_and_container_threaded(monkeypatch, no_sleep):
    core = SaveCore(cm_reads=[{"status": "Succeeded", "node": "n1"}])
    _patch_core(monkeypatch, core)
    sdk.save_image("j", tag="e:v1", team="t", worker=2, container="main")
    _, body = core.created[0]
    assert body["data"]["pod"] == "j-worker-2"
    assert body["data"]["container"] == "main"


def test_save_pod_missing_404_raises(monkeypatch, no_sleep):
    core = SaveCore(pod_exc=ApiException(status=404))
    _patch_core(monkeypatch, core)
    with pytest.raises(sdk.WujiError) as ei:
        sdk.save_image("j", tag="e:v1", team="t")
    assert "找不到容器" in str(ei.value)
    assert core.created == []  # no request left behind


def test_save_pod_other_api_error_raises(monkeypatch, no_sleep):
    core = SaveCore(pod_exc=ApiException(status=403))
    _patch_core(monkeypatch, core)
    with pytest.raises(sdk.WujiError):
        sdk.save_image("j", tag="e:v1", team="t")
    assert core.created == []


def test_save_pod_not_running_raises(monkeypatch, no_sleep):
    core = SaveCore(pod=_pod("Pending"))
    _patch_core(monkeypatch, core)
    with pytest.raises(sdk.WujiError) as ei:
        sdk.save_image("j", tag="e:v1", team="t")
    msg = str(ei.value)
    assert "Running" in msg
    assert core.created == []  # bailed before creating the request


def test_save_status_failed_raises_with_message(monkeypatch, no_sleep):
    core = SaveCore(cm_reads=[{"status": "Failed", "message": "commit boom"}])
    _patch_core(monkeypatch, core)
    with pytest.raises(sdk.WujiError) as ei:
        sdk.save_image("j", tag="e:v1", team="t")
    assert "commit boom" in str(ei.value)


def test_save_no_wait_returns_pending_without_polling(monkeypatch, no_sleep):
    core = SaveCore()
    _patch_core(monkeypatch, core)
    res = sdk.save_image("j", tag="e:v1", team="t", wait=False)
    assert res["status"] == "Pending"
    assert res["node"] == ""
    assert res["message"] == ""
    assert res["request"].startswith("wuji-save-j-w0-")
    assert core.read_calls == 0  # never polled


def test_save_timeout_raises(monkeypatch, no_sleep):
    # always-Pending index + a clock that steps past the deadline
    core = SaveCore(default_read={"status": "Pending"})
    _patch_core(monkeypatch, core)
    monkeypatch.setattr(sdk.time, "time", _Clock(start=0.0, step=5.0))
    with pytest.raises(sdk.WujiError) as ei:
        sdk.save_image("j", tag="e:v1", team="t", timeout=10, poll_interval=5)
    assert "超时" in str(ei.value)


def test_save_transient_read_error_keeps_polling(monkeypatch, no_sleep):
    # read raises once (transient) then returns Succeeded — must not crash
    seq = [ApiException(status=500), {"status": "Succeeded", "node": "n7"}]

    class FlakyCore(SaveCore):
        def read_namespaced_config_map(self, *, name, namespace):
            self.read_calls += 1
            item = seq.pop(0)
            if isinstance(item, ApiException):
                raise item
            return SimpleNamespace(data=dict(item))

    core = FlakyCore()
    _patch_core(monkeypatch, core)
    res = sdk.save_image("j", tag="e:v1", team="t")
    assert res["status"] == "Succeeded"
    assert res["node"] == "n7"
    assert core.read_calls == 2


# --------------------------------------------------------------------------- #
# list_images
# --------------------------------------------------------------------------- #
class IndexCore:
    def __init__(self, *, data=None, exc=None):
        self._data = data
        self._exc = exc
        self.calls = []

    def read_namespaced_config_map(self, *, name, namespace):
        self.calls.append((name, namespace))
        if self._exc:
            raise self._exc
        return SimpleNamespace(data=self._data)


def _index(monkeypatch, **kw):
    core = IndexCore(**kw)
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    return core


def test_list_images_index_404_returns_empty(monkeypatch):
    _index(monkeypatch, exc=ApiException(status=404))
    assert sdk.list_images() == []


def test_list_images_index_other_error_raises(monkeypatch):
    _index(monkeypatch, exc=ApiException(status=500))
    with pytest.raises(sdk.WujiError):
        sdk.list_images()


def test_list_images_parses_and_sorts(monkeypatch):
    data = {
        "b": json.dumps({"image": "saved-z:v1", "kind": "saved", "team": "t1"}),
        "a": json.dumps({"image": "base-a:v1", "kind": "base"}),
        "c": json.dumps({"image": "base-b:v1", "kind": "base", "created": "2026"}),
    }
    _index(monkeypatch, data=data)
    rows = sdk.list_images()
    # sorted by (kind, image): base-a, base-b, then saved-z
    assert [r["image"] for r in rows] == ["base-a:v1", "base-b:v1", "saved-z:v1"]
    assert [r["kind"] for r in rows] == ["base", "base", "saved"]
    assert rows[2]["team"] == "t1"
    assert rows[1]["created"] == "2026"


def test_list_images_kind_defaults_to_saved(monkeypatch):
    _index(monkeypatch, data={"x": json.dumps({"image": "e:v1"})})
    (row,) = sdk.list_images()
    assert row["kind"] == "saved"
    assert row["team"] == ""


def test_list_images_created_falls_back_to_ts(monkeypatch):
    _index(monkeypatch, data={"x": json.dumps({"image": "e:v1", "ts": "T0"})})
    (row,) = sdk.list_images()
    assert row["created"] == "T0"


def test_list_images_bad_json_falls_back_to_raw_image(monkeypatch):
    _index(monkeypatch, data={"legacy": "some/raw-image:v1"})
    (row,) = sdk.list_images()
    assert row["image"] == "some/raw-image:v1"
    assert row["kind"] == "saved"


def test_list_images_json_non_object_falls_back_to_raw(monkeypatch):
    # valid JSON but not a dict (e.g. a bare quoted string) → treat raw as image
    _index(monkeypatch, data={"weird": json.dumps(["not", "a", "dict"])})
    (row,) = sdk.list_images()
    assert row["image"] == json.dumps(["not", "a", "dict"])


def test_list_images_entry_without_image_skipped(monkeypatch):
    data = {
        "ok": json.dumps({"image": "e:v1"}),
        "noimg": json.dumps({"kind": "saved", "team": "t"}),
    }
    _index(monkeypatch, data=data)
    rows = sdk.list_images()
    assert [r["image"] for r in rows] == ["e:v1"]


def test_list_images_none_data_returns_empty(monkeypatch):
    _index(monkeypatch, data=None)
    assert sdk.list_images() == []


def test_list_images_team_filter_keeps_base_and_matching(monkeypatch):
    data = {
        "base": json.dumps({"image": "base:v1", "kind": "base"}),
        "mine": json.dumps({"image": "mine:v1", "kind": "saved", "team": "t1"}),
        "other": json.dumps({"image": "other:v1", "kind": "saved", "team": "t2"}),
    }
    _index(monkeypatch, data=data)
    rows = sdk.list_images(team="t1")
    imgs = {r["image"] for r in rows}
    assert imgs == {"base:v1", "mine:v1"}  # base always in; t2 filtered out


def test_list_images_non_string_kind_sorts_without_crash(monkeypatch):
    # LOW-7: sort key is (str(kind), str(image)) so non-string kinds don't crash
    data = {
        "a": json.dumps({"image": "e:v1", "kind": 123}),
        "b": json.dumps({"image": "f:v1", "kind": "base"}),
    }
    _index(monkeypatch, data=data)
    rows = sdk.list_images()
    # '123' < 'base' as strings
    assert [r["image"] for r in rows] == ["e:v1", "f:v1"]


def test_list_images_uses_default_index_location(monkeypatch):
    core = _index(monkeypatch, data={})
    sdk.list_images()
    name, ns = core.calls[0]
    assert (name, ns) == (sdk.IMAGE_INDEX_NAME, sdk.IMAGE_INDEX_NAMESPACE)
