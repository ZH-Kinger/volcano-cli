"""v0.9.0 SDK-layer tests: two-layer quotas + default-queue routing.

Covers the new ``volcano.sdk`` surface, fully mocked (no real cluster / kubeconfig):

* ``resolve_default_queue`` — reads the ns ``wuji.io/team`` label; fail-closed,
  no ``"default"`` fallback — no label, an unreadable namespace, a labelled
  team with no matching Queue, or any other lookup failure all raise
  ``WujiError`` instead of silently routing into the wide-open ``default``
  Queue (that used to let a submission duck team-level GPU accounting).
* ``get_quota`` / ``set_quota`` — per-person ResourceQuota ``gpu-quota``
  (admin-gated, create-then-patch upsert, clear, team-label write, report-only).
* ``get_queue_one`` / ``set_queue`` — per-team Volcano Queue upsert/delete
  (admin-gated, partial-field merge, reclaimable default, resourceVersion on
  replace).

Mocking approach: monkeypatch ``sdk.load_clients`` / ``sdk.current_namespace``
and swap the kubernetes ``client`` module object on ``volcano.sdk`` so the
``client.CoreV1Api()`` / ``AuthorizationV1Api()`` calls inside the functions
return our fakes. ``sdk._require_admin`` is stubbed to a no-op for the happy
paths and exercised for real (SSAR-denied) in the gate tests. Only tests/ is
touched.
"""

from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

import volcano.sdk as sdk
from volcano.kube import TEAM_LABEL, WujiError


# --------------------------------------------------------------------------- #
# admin gate fakes (real _require_admin path)
# --------------------------------------------------------------------------- #
class _Auth:
    """SelfSubjectAccessReview responder.

    ``allowed`` is a bool applied to every check, or a list consumed in order
    (per-check). ``exc`` raises instead of answering.
    """

    def __init__(self, allowed=True, exc=None):
        self.allowed = allowed
        self.exc = exc
        self.reviews = []

    def create_self_subject_access_review(self, review):
        self.reviews.append(review)
        if self.exc is not None:
            raise self.exc
        if isinstance(self.allowed, list):
            ok = self.allowed[len(self.reviews) - 1]
        else:
            ok = self.allowed
        return SimpleNamespace(status=SimpleNamespace(allowed=ok))


def _client(*, core=None, custom=None, auth=None):
    """Stand-in for the ``kubernetes.client`` module object used inside sdk."""
    return SimpleNamespace(
        CoreV1Api=lambda: core,
        CustomObjectsApi=lambda: custom,
        AuthorizationV1Api=lambda: auth,
        V1SelfSubjectAccessReview=lambda **kw: SimpleNamespace(**kw),
        V1SelfSubjectAccessReviewSpec=lambda **kw: SimpleNamespace(**kw),
        V1ResourceAttributes=lambda **kw: SimpleNamespace(**kw),
    )


# =========================================================================== #
# resolve_default_queue — reads the ns team label; fail-closed, no default fallback
# =========================================================================== #
class _NsCore:
    def __init__(self, labels=None, exc=None):
        self.labels = labels
        self.exc = exc
        self.read = []

    def read_namespace(self, name):
        self.read.append(name)
        if self.exc:
            raise self.exc
        return SimpleNamespace(metadata=SimpleNamespace(labels=self.labels))


class _QueueGetCustom:
    """Fake CustomObjectsApi for resolve_default_queue's queue-existence check.

    The hardened resolve_default_queue only routes to the team queue when a Queue
    by that name actually exists — it does a ``get_cluster_custom_object`` and only
    returns the team on success. ``exists=True`` → the get returns an object;
    ``exc`` → the get raises it (e.g. 404 ApiException = queue absent). Every call
    is recorded in ``gets`` so tests can assert it was (or wasn't) consulted.
    """

    def __init__(self, exists=True, exc=None):
        self.exists = exists
        self.exc = exc
        self.gets = []

    def get_cluster_custom_object(self, **kw):
        self.gets.append(kw)
        if self.exc is not None:
            raise self.exc
        return {"metadata": {"name": kw.get("name")}}


def _wire_resolve(monkeypatch, core, ns="alice", custom=None):
    monkeypatch.setattr(sdk, "current_namespace", lambda team=None: ns)
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, custom))


def test_resolve_default_queue_label_present(monkeypatch):
    # label=team AND a Queue named after it exists -> route to the team queue.
    core = _NsCore(labels={TEAM_LABEL: "redteam", "other": "x"})
    custom = _QueueGetCustom(exists=True)
    _wire_resolve(monkeypatch, core, custom=custom)
    assert sdk.resolve_default_queue() == "redteam"
    assert core.read == ["alice"]  # read the resolved ns
    assert custom.gets[0]["name"] == "redteam"  # validated queue existence


def test_resolve_default_queue_team_arg_threaded(monkeypatch):
    core = _NsCore(labels={TEAM_LABEL: "blueteam"})
    custom = _QueueGetCustom(exists=True)
    seen = {}
    monkeypatch.setattr(sdk, "current_namespace", lambda team=None: seen.setdefault("t", team) or "bob")
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, custom))
    assert sdk.resolve_default_queue("bob") == "blueteam"
    assert seen["t"] == "bob"
    assert custom.gets[0]["name"] == "blueteam"


def test_resolve_default_queue_labelled_but_queue_absent_raises(monkeypatch):
    # label=team but NO such Queue (get 404) -> raise, never silently "default".
    core = _NsCore(labels={TEAM_LABEL: "orphanteam"})
    custom = _QueueGetCustom(exc=ApiException(status=404))
    _wire_resolve(monkeypatch, core, custom=custom)
    with pytest.raises(WujiError, match="orphanteam"):
        sdk.resolve_default_queue()
    assert custom.gets[0]["name"] == "orphanteam"  # it did try to validate


def test_resolve_default_queue_queue_get_non404_raises(monkeypatch):
    # any other ApiException from the queue get (e.g. 403/500) -> raise too.
    core = _NsCore(labels={TEAM_LABEL: "redteam"})
    custom = _QueueGetCustom(exc=ApiException(status=500))
    _wire_resolve(monkeypatch, core, custom=custom)
    with pytest.raises(WujiError):
        sdk.resolve_default_queue()


def test_resolve_default_queue_label_literally_default_still_probes(monkeypatch):
    # label == "default" is just a regular team name now — no special-cased skip;
    # it's validated like any other team and routes there if the Queue exists.
    core = _NsCore(labels={TEAM_LABEL: "default"})
    custom = _QueueGetCustom(exists=True)
    _wire_resolve(monkeypatch, core, custom=custom)
    assert sdk.resolve_default_queue() == "default"
    assert custom.gets[0]["name"] == "default"  # did consult the queue API


def test_resolve_default_queue_no_label_raises(monkeypatch):
    _wire_resolve(monkeypatch, _NsCore(labels={"unrelated": "y"}))
    with pytest.raises(WujiError, match="wuji.io/team"):
        sdk.resolve_default_queue()


def test_resolve_default_queue_empty_labels_raises(monkeypatch):
    _wire_resolve(monkeypatch, _NsCore(labels={}))
    with pytest.raises(WujiError):
        sdk.resolve_default_queue()


def test_resolve_default_queue_none_labels_raises(monkeypatch):
    # metadata.labels is None for a ns with no labels at all.
    _wire_resolve(monkeypatch, _NsCore(labels=None))
    with pytest.raises(WujiError):
        sdk.resolve_default_queue()


def test_resolve_default_queue_read_apiexception_raises(monkeypatch):
    _wire_resolve(monkeypatch, _NsCore(exc=ApiException(status=403)))
    with pytest.raises(WujiError):
        sdk.resolve_default_queue()


def test_resolve_default_queue_load_clients_raises_propagates(monkeypatch):
    monkeypatch.setattr(sdk, "current_namespace", lambda team=None: "carol")
    monkeypatch.setattr(
        sdk, "load_clients",
        lambda: (_ for _ in ()).throw(WujiError("no kubeconfig")),
    )
    with pytest.raises(WujiError, match="no kubeconfig"):
        sdk.resolve_default_queue()


def test_resolve_default_queue_current_namespace_raises_propagates(monkeypatch):
    # current_namespace can raise WujiError (admin with no default ns) — this now
    # propagates directly (no swallow-to-"default") since it blocks submit anyway
    # once there's no team to resolve.
    monkeypatch.setattr(
        sdk, "current_namespace",
        lambda team=None: (_ for _ in ()).throw(WujiError("无法确定团队 namespace")),
    )
    monkeypatch.setattr(
        sdk, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not run")),
    )
    with pytest.raises(WujiError, match="无法确定团队 namespace"):
        sdk.resolve_default_queue()


# =========================================================================== #
# get_quota
# =========================================================================== #
class _QuotaCore:
    def __init__(self, *, read_obj=None, read_exc=None, create_exc=None, delete_exc=None):
        self.read_obj = read_obj
        self.read_exc = read_exc
        self.create_exc = create_exc
        self.delete_exc = delete_exc
        self.calls = []
        self.created_body = None
        self.patched_body = None

    def names(self):
        return [c[0] for c in self.calls]

    def patch_namespace(self, name, body):
        self.calls.append(("patch_namespace", name, body))

    def create_namespaced_resource_quota(self, namespace, body):
        self.calls.append(("create_rq", namespace, body))
        self.created_body = body
        if self.create_exc:
            raise self.create_exc

    def patch_namespaced_resource_quota(self, name, namespace, body):
        self.calls.append(("patch_rq", name, namespace, body))
        self.patched_body = body

    def delete_namespaced_resource_quota(self, name, namespace):
        self.calls.append(("delete_rq", name, namespace))
        if self.delete_exc:
            raise self.delete_exc

    def read_namespaced_resource_quota(self, name, namespace):
        self.calls.append(("read_rq", name, namespace))
        if self.read_exc:
            raise self.read_exc
        if self.read_obj is not None:
            return self.read_obj
        raise ApiException(status=404)


class _FakeAssignCustom:
    """Minimal CustomObjectsApi stand-in for the Gatekeeper Assign calls used by
    get_home_node/set_home_node/clear_home_node. Defaults to "nothing set" (404
    on read) unless seeded via ``existing``."""

    def __init__(self, *, existing=None):
        self.existing = dict(existing or {})
        self.calls = []

    def get_cluster_custom_object(self, *, group, version, plural, name):
        self.calls.append(("get", name))
        if name in self.existing:
            return self.existing[name]
        raise ApiException(status=404)

    def create_cluster_custom_object(self, *, group, version, plural, body):
        name = body["metadata"]["name"]
        self.calls.append(("create", name))
        if name in self.existing:
            raise ApiException(status=409)
        self.existing[name] = body

    def patch_cluster_custom_object(self, *, group, version, plural, name, body):
        self.calls.append(("patch", name))
        entry = self.existing.setdefault(name, {"metadata": {"name": name}})
        entry.update(body)

    def delete_cluster_custom_object(self, *, group, version, plural, name):
        self.calls.append(("delete", name))
        if name not in self.existing:
            raise ApiException(status=404)
        del self.existing[name]


def _rq(hard, used):
    # get_quota reads the limit from spec.hard (visible immediately after
    # create/patch) and usage from status.used (filled in by the quota controller).
    return SimpleNamespace(
        spec=SimpleNamespace(hard=hard),
        status=SimpleNamespace(used=used),
    )


def test_get_quota_returns_hard_and_used(monkeypatch):
    core = _QuotaCore(read_obj=_rq(
        {"requests.nvidia.com/gpu": "8", "pods": "20"},
        {"requests.nvidia.com/gpu": "3", "pods": "5"},
    ))
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    out = sdk.get_quota("alice")
    assert out == {
        "namespace": "alice",
        "hard": {"requests.nvidia.com/gpu": "8", "pods": "20"},
        "used": {"requests.nvidia.com/gpu": "3", "pods": "5"},
    }


def test_get_quota_404_returns_none(monkeypatch):
    core = _QuotaCore(read_exc=ApiException(status=404))
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    assert sdk.get_quota("alice") is None


def test_get_quota_other_error_raises_wuji(monkeypatch):
    core = _QuotaCore(read_exc=ApiException(status=500))
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    with pytest.raises(WujiError):
        sdk.get_quota("alice")


def test_get_quota_none_spec_status_maps_empty(monkeypatch):
    core = _QuotaCore(read_obj=SimpleNamespace(spec=None, status=None))
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    out = sdk.get_quota("alice")
    assert out == {"namespace": "alice", "hard": {}, "used": {}}


# =========================================================================== #
# set_quota
# =========================================================================== #
def _wire_quota_admin(monkeypatch, core, assign_custom=None):
    """Admin path: _require_admin is a no-op; client.CoreV1Api() -> core."""
    custom = assign_custom if assign_custom is not None else _FakeAssignCustom()
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, custom))
    monkeypatch.setattr(sdk, "_require_admin", lambda: None)
    monkeypatch.setattr(sdk, "client", _client(core=core))


def test_set_quota_requires_admin_denied(monkeypatch):
    # Real _require_admin with SSAR denied -> WujiError, before any RQ write.
    core = _QuotaCore()
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "client", _client(core=core, auth=_Auth(allowed=False)))
    with pytest.raises(WujiError) as ei:
        sdk.set_quota("alice", gpu=8)
    assert "管理员" in str(ei.value)
    assert "create_rq" not in core.names() and "patch_rq" not in core.names()


def test_set_quota_require_admin_is_invoked(monkeypatch):
    called = {"n": 0}
    core = _QuotaCore()
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, _FakeAssignCustom()))
    monkeypatch.setattr(sdk, "_require_admin", lambda: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setattr(sdk, "client", _client(core=core))
    sdk.set_quota("alice", gpu=8)
    assert called["n"] == 1


def test_set_quota_gpu_maps_requests_and_limits(monkeypatch):
    core = _QuotaCore()
    _wire_quota_admin(monkeypatch, core)
    sdk.set_quota("alice", gpu=8)
    hard = core.created_body["spec"]["hard"]
    assert hard["requests.nvidia.com/gpu"] == "8"
    assert hard["limits.nvidia.com/gpu"] == "8"
    assert core.created_body["metadata"]["name"] == "gpu-quota"


def test_set_quota_cpu_memory_pods_mapping(monkeypatch):
    # cpu/memory map to both requests and limits (same value) — mirrors the GPU
    # dimension's requests==limits semantics, so the admin-set ceiling is a real
    # cgroup limit, not just a scheduling-time reservation a pod can burst past.
    core = _QuotaCore()
    _wire_quota_admin(monkeypatch, core)
    sdk.set_quota("alice", cpu="64", memory="256Gi", pods=10)
    hard = core.created_body["spec"]["hard"]
    assert hard == {
        "requests.cpu": "64",
        "limits.cpu": "64",
        "requests.memory": "256Gi",
        "limits.memory": "256Gi",
        "pods": "10",
    }


def test_set_quota_creates_when_absent(monkeypatch):
    core = _QuotaCore()  # create succeeds
    _wire_quota_admin(monkeypatch, core)
    sdk.set_quota("alice", gpu=4)
    assert "create_rq" in core.names()
    assert "patch_rq" not in core.names()


def test_set_quota_patches_on_conflict(monkeypatch):
    core = _QuotaCore(create_exc=ApiException(status=409))
    _wire_quota_admin(monkeypatch, core)
    sdk.set_quota("alice", gpu=4)
    assert "create_rq" in core.names()
    assert "patch_rq" in core.names()
    assert core.patched_body == {"spec": {"hard": {
        "requests.nvidia.com/gpu": "4", "limits.nvidia.com/gpu": "4"
    }}}


def test_set_quota_create_other_error_raises(monkeypatch):
    core = _QuotaCore(create_exc=ApiException(status=500))
    _wire_quota_admin(monkeypatch, core)
    with pytest.raises(WujiError):
        sdk.set_quota("alice", gpu=4)


def test_set_quota_clear_deletes(monkeypatch):
    core = _QuotaCore()
    _wire_quota_admin(monkeypatch, core)
    out = sdk.set_quota("alice", clear=True)
    assert "delete_rq" in core.names()
    assert out["cleared"] is True and out["quota"] is None
    # clear must NOT create/patch a quota
    assert "create_rq" not in core.names() and "patch_rq" not in core.names()


def test_set_quota_clear_swallows_404(monkeypatch):
    core = _QuotaCore(delete_exc=ApiException(status=404))
    _wire_quota_admin(monkeypatch, core)
    out = sdk.set_quota("alice", clear=True)  # no raise
    assert out["cleared"] is True


def test_set_quota_clear_other_error_raises(monkeypatch):
    core = _QuotaCore(delete_exc=ApiException(status=500))
    _wire_quota_admin(monkeypatch, core)
    with pytest.raises(WujiError):
        sdk.set_quota("alice", clear=True)


def test_set_quota_team_writes_namespace_label(monkeypatch):
    core = _QuotaCore()
    _wire_quota_admin(monkeypatch, core)
    out = sdk.set_quota("alice", gpu=8, team="redteam")
    patch = [c for c in core.calls if c[0] == "patch_namespace"]
    assert len(patch) == 1
    _, name, body = patch[0]
    assert name == "alice"
    assert body["metadata"]["labels"][TEAM_LABEL] == "redteam"
    assert out["team"] == "redteam"


def test_set_quota_team_only_no_limits_reports_current(monkeypatch):
    # team given but no limit flags: writes label, does NOT create/patch a quota,
    # returns get_quota(namespace).
    core = _QuotaCore(read_exc=ApiException(status=404))
    _wire_quota_admin(monkeypatch, core)
    out = sdk.set_quota("alice", team="blue")
    assert "patch_namespace" in core.names()
    assert "create_rq" not in core.names() and "patch_rq" not in core.names()
    assert out["team"] == "blue"
    assert out["quota"] is None  # get_quota hit a 404


def test_set_quota_no_args_reports_only(monkeypatch):
    core = _QuotaCore(read_obj=_rq({"pods": "5"}, {"pods": "1"}))
    _wire_quota_admin(monkeypatch, core)
    out = sdk.set_quota("alice")
    assert "create_rq" not in core.names() and "patch_rq" not in core.names()
    assert out["team"] is None
    assert out["quota"]["hard"] == {"pods": "5"}


# =========================================================================== #
# home-node soft affinity (get/set/clear_home_node) — punchlist §8
# =========================================================================== #
def _assign_body(namespace, node):
    from volcano.sdk import _home_node_assign_name, _home_node_assign_spec
    spec = _home_node_assign_spec(node)
    spec["match"]["namespaces"] = [namespace]
    return {"metadata": {"name": _home_node_assign_name(namespace)}, "spec": spec}


def test_get_home_node_none_when_absent(monkeypatch):
    custom = _FakeAssignCustom()
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    assert sdk.get_home_node("alice") is None


def test_get_home_node_returns_set_value(monkeypatch):
    custom = _FakeAssignCustom(existing={"home-node-alice": _assign_body("alice", "wuji-2")})
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    assert sdk.get_home_node("alice") == "wuji-2"


def test_get_home_node_malformed_returns_none(monkeypatch):
    custom = _FakeAssignCustom(existing={"home-node-alice": {"metadata": {"name": "home-node-alice"}, "spec": {}}})
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    assert sdk.get_home_node("alice") is None


def test_get_home_node_other_error_raises(monkeypatch):
    class Boom:
        def get_cluster_custom_object(self, **kw):
            raise ApiException(status=500)

    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, Boom()))
    with pytest.raises(WujiError):
        sdk.get_home_node("alice")


def test_set_home_node_creates_when_absent(monkeypatch):
    custom = _FakeAssignCustom()
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    sdk.set_home_node("alice", "wuji-3")
    assert custom.calls == [("create", "home-node-alice")]
    assert sdk.get_home_node("alice") == "wuji-3"


def test_set_home_node_patches_on_conflict(monkeypatch):
    custom = _FakeAssignCustom(existing={"home-node-alice": _assign_body("alice", "wuji-0")})
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    sdk.set_home_node("alice", "wuji-3")
    assert ("create", "home-node-alice") in custom.calls
    assert ("patch", "home-node-alice") in custom.calls
    assert sdk.get_home_node("alice") == "wuji-3"


def test_set_home_node_pathtests_must_not_exist():
    # so a user's own pod-spec affinity isn't clobbered by the default.
    from volcano.sdk import _home_node_assign_spec
    spec = _home_node_assign_spec("wuji-1")
    assert spec["parameters"]["pathTests"] == [{
        "subPath": "spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution",
        "condition": "MustNotExist",
    }]
    pref = spec["parameters"]["assign"]["value"][0]
    assert pref["weight"] == 100
    assert pref["preference"]["matchExpressions"][0]["values"] == ["wuji-1"]


def test_clear_home_node_deletes_and_returns_true(monkeypatch):
    custom = _FakeAssignCustom(existing={"home-node-alice": _assign_body("alice", "wuji-1")})
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    assert sdk.clear_home_node("alice") is True
    assert sdk.get_home_node("alice") is None


def test_clear_home_node_absent_returns_false(monkeypatch):
    custom = _FakeAssignCustom()
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    assert sdk.clear_home_node("alice") is False


# --- set_quota integration: --home-node/--no-home-node alongside quota/team --- #
def test_set_quota_home_node_sets_assign(monkeypatch):
    core = _QuotaCore()
    custom = _FakeAssignCustom()
    _wire_quota_admin(monkeypatch, core, assign_custom=custom)
    out = sdk.set_quota("alice", home_node="wuji-2")
    assert out["home_node"] == "wuji-2"
    assert sdk.get_home_node("alice") == "wuji-2"


def test_set_quota_no_home_node_clears_assign(monkeypatch):
    core = _QuotaCore()
    custom = _FakeAssignCustom(existing={"home-node-alice": _assign_body("alice", "wuji-2")})
    _wire_quota_admin(monkeypatch, core, assign_custom=custom)
    out = sdk.set_quota("alice", no_home_node=True)
    assert out["home_node"] is None


def test_set_quota_home_node_and_no_home_node_conflict(monkeypatch):
    core = _QuotaCore()
    _wire_quota_admin(monkeypatch, core)
    with pytest.raises(WujiError, match="只能给一个"):
        sdk.set_quota("alice", home_node="wuji-2", no_home_node=True)


def test_set_quota_reports_current_home_node_when_untouched(monkeypatch):
    core = _QuotaCore()
    custom = _FakeAssignCustom(existing={"home-node-alice": _assign_body("alice", "wuji-4")})
    _wire_quota_admin(monkeypatch, core, assign_custom=custom)
    out = sdk.set_quota("alice", gpu=8)
    assert out["home_node"] == "wuji-4"


def test_set_quota_home_node_independent_of_team_and_clear(monkeypatch):
    # --team X --home-node Y --clear in one call: all three take effect.
    core = _QuotaCore()
    custom = _FakeAssignCustom()
    _wire_quota_admin(monkeypatch, core, assign_custom=custom)
    out = sdk.set_quota("alice", team="redteam", home_node="wuji-1", clear=True)
    assert out["team"] == "redteam"
    assert out["home_node"] == "wuji-1"
    assert out["cleared"] is True


# =========================================================================== #
# get_queue_one
# =========================================================================== #
class _QueueCustom:
    def __init__(self, existing=None, get_exc=None):
        self.existing = existing
        self.get_exc = get_exc
        self.calls = []
        self.created_body = None
        self.replaced_body = None

    def names(self):
        return [c[0] for c in self.calls]

    def get_cluster_custom_object(self, **kw):
        self.calls.append(("get", kw))
        if self.get_exc is not None:
            raise self.get_exc
        if self.existing is None:
            raise ApiException(status=404)
        return self.existing

    def create_cluster_custom_object(self, **kw):
        self.calls.append(("create", kw))
        self.created_body = kw["body"]
        self.existing = kw["body"]  # now readable by the trailing get_queue_one

    def replace_cluster_custom_object(self, **kw):
        self.calls.append(("replace", kw))
        self.replaced_body = kw["body"]
        self.existing = kw["body"]

    def delete_cluster_custom_object(self, **kw):
        self.calls.append(("delete", kw))
        self.existing = None


def test_get_queue_one_reads_fields(monkeypatch):
    obj = {
        "spec": {
            "deserved": {"nvidia.com/gpu": "4"},
            "capability": {"nvidia.com/gpu": "8"},
            "reclaimable": False,
        },
        "status": {"state": "Open"},
    }
    custom = _QueueCustom(existing=obj)
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    out = sdk.get_queue_one("redteam")
    assert out == {
        "name": "redteam",
        "deserved": "4",
        "capability": "8",
        "reclaimable": False,
        "state": "Open",
    }


def test_get_queue_one_404_returns_none(monkeypatch):
    custom = _QueueCustom(existing=None)
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    assert sdk.get_queue_one("ghost") is None


def test_get_queue_one_missing_fields_default_dash(monkeypatch):
    custom = _QueueCustom(existing={"spec": {}, "status": {}})
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    out = sdk.get_queue_one("t")
    assert out["deserved"] == "-" and out["capability"] == "-" and out["state"] == "-"


def test_get_queue_one_other_error_raises(monkeypatch):
    custom = _QueueCustom(get_exc=ApiException(status=500))
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    with pytest.raises(WujiError):
        sdk.get_queue_one("t")


# =========================================================================== #
# set_queue
# =========================================================================== #
def _wire_queue_admin(monkeypatch, custom):
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    monkeypatch.setattr(sdk, "_require_admin", lambda: None)
    monkeypatch.setattr(sdk, "client", _client(custom=custom))


def test_set_queue_requires_admin_denied(monkeypatch):
    custom = _QueueCustom()
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    monkeypatch.setattr(sdk, "client", _client(custom=custom, auth=_Auth(allowed=False)))
    with pytest.raises(WujiError):
        sdk.set_queue("redteam", deserved=4)
    assert "create" not in custom.names() and "replace" not in custom.names()


def test_set_queue_require_admin_is_invoked(monkeypatch):
    called = {"n": 0}
    custom = _QueueCustom()
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    monkeypatch.setattr(sdk, "_require_admin", lambda: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setattr(sdk, "client", _client(custom=custom))
    sdk.set_queue("redteam", deserved=4)
    assert called["n"] == 1


def test_set_queue_creates_when_absent_default_reclaimable_true(monkeypatch):
    custom = _QueueCustom(existing=None)
    _wire_queue_admin(monkeypatch, custom)
    out = sdk.set_queue("redteam", deserved=4, cap=8)
    assert "create" in custom.names() and "replace" not in custom.names()
    spec = custom.created_body["spec"]
    assert spec["deserved"]["nvidia.com/gpu"] == "4"
    assert spec["capability"]["nvidia.com/gpu"] == "8"
    assert spec["reclaimable"] is True  # new queues default reclaimable
    assert custom.created_body["kind"] == "Queue"
    assert custom.created_body["metadata"]["name"] == "redteam"
    assert out["created"] is True


def test_set_queue_replaces_existing_with_resource_version(monkeypatch):
    existing = {
        "metadata": {"name": "redteam", "resourceVersion": "12345"},
        "spec": {"deserved": {"nvidia.com/gpu": "2"},
                 "capability": {"nvidia.com/gpu": "9"}, "reclaimable": True},
    }
    custom = _QueueCustom(existing=existing)
    _wire_queue_admin(monkeypatch, custom)
    out = sdk.set_queue("redteam", deserved=6)
    assert "replace" in custom.names() and "create" not in custom.names()
    body = custom.replaced_body
    assert body["metadata"]["resourceVersion"] == "12345"
    assert out["created"] is False


def test_set_queue_only_deserved_leaves_cap(monkeypatch):
    existing = {
        "metadata": {"name": "redteam", "resourceVersion": "1"},
        "spec": {"deserved": {"nvidia.com/gpu": "2"},
                 "capability": {"nvidia.com/gpu": "9"}, "reclaimable": True},
    }
    custom = _QueueCustom(existing=existing)
    _wire_queue_admin(monkeypatch, custom)
    sdk.set_queue("redteam", deserved=5)
    spec = custom.replaced_body["spec"]
    assert spec["deserved"]["nvidia.com/gpu"] == "5"
    assert spec["capability"]["nvidia.com/gpu"] == "9"  # untouched


def test_set_queue_only_cap_leaves_deserved(monkeypatch):
    existing = {
        "metadata": {"name": "redteam", "resourceVersion": "1"},
        "spec": {"deserved": {"nvidia.com/gpu": "2"},
                 "capability": {"nvidia.com/gpu": "9"}, "reclaimable": True},
    }
    custom = _QueueCustom(existing=existing)
    _wire_queue_admin(monkeypatch, custom)
    sdk.set_queue("redteam", cap=16)
    spec = custom.replaced_body["spec"]
    assert spec["capability"]["nvidia.com/gpu"] == "16"
    assert spec["deserved"]["nvidia.com/gpu"] == "2"  # untouched


def test_set_queue_reclaimable_false_passthrough(monkeypatch):
    custom = _QueueCustom(existing=None)
    _wire_queue_admin(monkeypatch, custom)
    sdk.set_queue("redteam", deserved=4, reclaimable=False)
    assert custom.created_body["spec"]["reclaimable"] is False


def test_set_queue_reclaimable_true_passthrough_on_existing(monkeypatch):
    existing = {
        "metadata": {"name": "redteam", "resourceVersion": "1"},
        "spec": {"reclaimable": False},
    }
    custom = _QueueCustom(existing=existing)
    _wire_queue_admin(monkeypatch, custom)
    sdk.set_queue("redteam", reclaimable=True)
    assert custom.replaced_body["spec"]["reclaimable"] is True


def test_set_queue_delete(monkeypatch):
    custom = _QueueCustom(existing={"metadata": {"name": "redteam"}, "spec": {}})
    _wire_queue_admin(monkeypatch, custom)
    out = sdk.set_queue("redteam", delete=True)
    assert out == {"name": "redteam", "deleted": True}
    assert "delete" in custom.names()
    # delete short-circuits: no get/create/replace
    assert "create" not in custom.names() and "replace" not in custom.names()


def test_set_queue_delete_api_error_raises(monkeypatch):
    class _C(_QueueCustom):
        def delete_cluster_custom_object(self, **kw):
            raise ApiException(status=403)

    custom = _C()
    _wire_queue_admin(monkeypatch, custom)
    with pytest.raises(WujiError):
        sdk.set_queue("redteam", delete=True)


def test_set_queue_get_other_error_raises(monkeypatch):
    custom = _QueueCustom(get_exc=ApiException(status=500))
    _wire_queue_admin(monkeypatch, custom)
    with pytest.raises(WujiError):
        sdk.set_queue("redteam", deserved=4)


def test_set_queue_no_args_view_only(monkeypatch):
    # No change flags -> view only: no admin gate, no create/replace, no
    # "created" key in the result. queue is None when the queue doesn't exist.
    custom = _QueueCustom(existing=None)
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    monkeypatch.setattr(
        sdk, "_require_admin",
        lambda: (_ for _ in ()).throw(AssertionError("view mode must not require admin")),
    )
    out = sdk.set_queue("redteam")
    assert out == {"name": "redteam", "queue": None}
    assert "created" not in out
    assert "create" not in custom.names() and "replace" not in custom.names()


def test_set_queue_no_args_view_existing(monkeypatch):
    obj = {
        "spec": {"deserved": {"nvidia.com/gpu": "4"},
                 "capability": {"nvidia.com/gpu": "8"}, "reclaimable": True},
        "status": {"state": "Open"},
    }
    custom = _QueueCustom(existing=obj)
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, custom))
    monkeypatch.setattr(
        sdk, "_require_admin",
        lambda: (_ for _ in ()).throw(AssertionError("view mode must not require admin")),
    )
    out = sdk.set_queue("redteam")
    assert "created" not in out
    assert out["queue"]["deserved"] == "4" and out["queue"]["state"] == "Open"


# =========================================================================== #
# check_admin — SSAR proxy for "is admin"; non-raising (any failure -> False)
# =========================================================================== #
def test_check_admin_all_allowed_true(monkeypatch):
    auth = _Auth(allowed=True)
    monkeypatch.setattr(sdk, "client", _client(auth=auth))
    assert sdk.check_admin() is True
    assert len(auth.reviews) == 2  # namespaces + clusterrolebindings both checked


def test_check_admin_first_denied_false(monkeypatch):
    monkeypatch.setattr(sdk, "client", _client(auth=_Auth(allowed=[False, True])))
    assert sdk.check_admin() is False


def test_check_admin_second_denied_false(monkeypatch):
    monkeypatch.setattr(sdk, "client", _client(auth=_Auth(allowed=[True, False])))
    assert sdk.check_admin() is False


def test_check_admin_api_error_false(monkeypatch):
    monkeypatch.setattr(sdk, "client", _client(auth=_Auth(exc=ApiException(status=500))))
    assert sdk.check_admin() is False


# =========================================================================== #
# whoami — identity assembly (best-effort; never raises on missing ns/label)
# =========================================================================== #
def test_whoami_admin_with_team(monkeypatch):
    core = _NsCore(labels={TEAM_LABEL: "redteam"})
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "client", _client(core=core, auth=_Auth(allowed=True)))
    monkeypatch.setattr(sdk, "current_namespace", lambda team=None: "alice")
    out = sdk.whoami()
    assert out == {
        "namespace": "alice", "is_admin": True,
        "team": "redteam", "default_queue": "redteam",
    }


def test_whoami_non_admin_no_team(monkeypatch):
    core = _NsCore(labels={})
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "client", _client(core=core, auth=_Auth(allowed=False)))
    monkeypatch.setattr(sdk, "current_namespace", lambda team=None: "bob")
    out = sdk.whoami()
    assert out["is_admin"] is False
    assert out["team"] is None
    assert out["default_queue"] == "default"


def test_whoami_no_namespace(monkeypatch):
    # current_namespace raises (admin kubeconfig w/o default ns) -> ns None,
    # team None, default_queue "default"; read_namespace must not run.
    core = _NsCore(exc=AssertionError("read_namespace must not run without ns"))
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "client", _client(core=core, auth=_Auth(allowed=True)))
    monkeypatch.setattr(
        sdk, "current_namespace",
        lambda team=None: (_ for _ in ()).throw(WujiError("无法确定团队 namespace")),
    )
    out = sdk.whoami()
    assert out["namespace"] is None
    assert out["team"] is None
    assert out["default_queue"] == "default"
    assert out["is_admin"] is True


def test_whoami_read_namespace_fails_team_none(monkeypatch):
    core = _NsCore(exc=ApiException(status=403))
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "client", _client(core=core, auth=_Auth(allowed=False)))
    monkeypatch.setattr(sdk, "current_namespace", lambda team=None: "alice")
    out = sdk.whoami()
    assert out["namespace"] == "alice"
    assert out["team"] is None
    assert out["default_queue"] == "default"
