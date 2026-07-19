"""v0.10.0 tests: ``volcano grant-admin`` (admin makes someone a platform admin).

Fully mocked — no real cluster / kubeconfig. Two layers:

* SDK ``sdk.grant_admin(namespace, *, reason, sa=None, revoke=False)``:
  ``load_clients`` → ``_require_admin`` gate → ``RbacAuthorizationV1Api``. Binds
  ``<ns>``'s SA to the built-in ``cluster-admin`` ClusterRole via a
  ClusterRoleBinding ``<ns>-admin`` (idempotent if already bound to it; error if
  bound to a DIFFERENT role — roleRef is immutable, revoke first). ``--revoke``
  deletes the binding (404 swallowed). There is only one admin tier — punchlist
  §10: a scoped-down "volcano-admin" ClusterRole used to exist here, but the
  user explicitly rejected that design ("两种 admin 就应该是一样的，admin 就应该
  有所有权限才对"), so grant_admin() no longer creates/patches any custom
  ClusterRole, it only ever binds the built-in ``cluster-admin``.
  ``reason`` is mandatory (blank → WujiError, before any API call); every call
  (grant, revoke, or idempotent re-grant) appends an entry — who/whom/when/why —
  to ``wuji-system/admin-grant-audit-log`` (punchlist §4: grant-admin used to
  change hands silently). The audit write is best-effort: a failure there
  degrades the result to ``audit_logged: False`` rather than raising, since the
  RBAC change has already happened by that point.
* CLI ``grant-admin`` (grant_admin_cmd): threads namespace/--reason/--sa/
  --revoke into ``sdk.grant_admin`` and prints friendly lines; ``--reason`` is a
  required typer option; a ``WujiError`` → non-zero exit; an
  ``audit_logged: False`` result prints a warning but doesn't fail the command.

Mocking approach mirrors tests/test_v090_sdk_quota_queue.py: monkeypatch
``sdk.load_clients`` / ``sdk._require_admin`` and swap the kubernetes ``client``
module object on ``volcano.sdk`` so ``client.RbacAuthorizationV1Api()`` (and, for the
real-gate test, ``client.AuthorizationV1Api()`` + SSAR builders) return our fakes.
``load_clients`` also hands back a fake ``CoreV1Api``-ish object (``_AuditCore``)
standing in for the audit ConfigMap's read/patch/create calls, and
``client.AuthenticationV1Api()`` a fake SelfSubjectReview responder for the
operator-identity attribution. Only tests/ is touched.
"""

import json
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException
from typer.testing import CliRunner

import volcano.cli as cli
import volcano.sdk as sdk
from volcano.kube import WujiError

runner = CliRunner()


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _Auth:
    """SelfSubjectAccessReview responder used by the real ``_require_admin`` path."""

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


class _FakeSelfSubjectReview:
    """AuthenticationV1Api().create_self_subject_review() responder — drives the
    audit entry's "operator" attribution. ``exc`` forces the "unknown" fallback."""

    def __init__(self, username="test-operator", exc=None):
        self.username = username
        self.exc = exc

    def create_self_subject_review(self, body):
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(status=SimpleNamespace(user_info=SimpleNamespace(username=self.username)))


class _AuditCore:
    """Fake CoreV1Api standing in for the admin-grant-audit-log ConfigMap calls.

    ``existing_log=None`` (default) means the ConfigMap doesn't exist yet ->
    read 404s -> create path. Seed a list to simulate an existing log -> read
    succeeds -> patch (append) path. ``*_exc`` forces a failure at that step.
    """

    def __init__(self, *, existing_log=None, read_exc=None, patch_exc=None, create_exc=None):
        self.existing_log = existing_log
        self.read_exc = read_exc
        self.patch_exc = patch_exc
        self.create_exc = create_exc
        self.patched = None    # (name, namespace, body)
        self.created = None    # body

    def read_namespaced_config_map(self, name, namespace):
        if self.read_exc is not None:
            raise self.read_exc
        if self.existing_log is None:
            raise ApiException(status=404)
        return SimpleNamespace(data={"log": json.dumps(self.existing_log, ensure_ascii=False)})

    def patch_namespaced_config_map(self, name, namespace, body):
        if self.patch_exc is not None:
            raise self.patch_exc
        self.patched = (name, namespace, body)

    def create_namespaced_config_map(self, namespace, body):
        if self.create_exc is not None:
            raise self.create_exc
        self.created = body


def _binding(role):
    """A fake existing ClusterRoleBinding whose roleRef points at ``role``."""
    return SimpleNamespace(role_ref=SimpleNamespace(name=role))


class _Rbac:
    """Records RBAC calls; each ``*_exc`` raises from the matching call.

    ``read_obj`` / ``read_exc`` drive ``read_cluster_role_binding``: an object to
    return (existing binding), an exception to raise, else a default 404 (absent).
    """

    def __init__(self, *, create_crb_exc=None, delete_crb_exc=None,
                 read_obj=None, read_exc=None):
        self.create_crb_exc = create_crb_exc
        self.delete_crb_exc = delete_crb_exc
        self.read_obj = read_obj
        self.read_exc = read_exc
        self.calls = []
        self.created_crb = None
        self.deleted_crb = None         # name
        self.read_crb = None            # name

    def names(self):
        return [c[0] for c in self.calls]

    def read_cluster_role_binding(self, name):
        self.calls.append(("read_cluster_role_binding", name))
        self.read_crb = name
        if self.read_exc is not None:
            raise self.read_exc
        if self.read_obj is not None:
            return self.read_obj
        raise ApiException(status=404)  # absent by default -> create path

    def create_cluster_role_binding(self, body):
        self.calls.append(("create_cluster_role_binding", body))
        self.created_crb = body
        if self.create_crb_exc:
            raise self.create_crb_exc

    def delete_cluster_role_binding(self, name):
        self.calls.append(("delete_cluster_role_binding", name))
        self.deleted_crb = name
        if self.delete_crb_exc:
            raise self.delete_crb_exc


def _client(*, rbac=None, auth=None, self_review=None):
    """Stand-in for the ``kubernetes.client`` module object used inside sdk."""
    return SimpleNamespace(
        RbacAuthorizationV1Api=lambda: rbac,
        AuthorizationV1Api=lambda: auth,
        AuthenticationV1Api=lambda: (self_review if self_review is not None else _FakeSelfSubjectReview()),
        V1SelfSubjectAccessReview=lambda **kw: SimpleNamespace(**kw),
        V1SelfSubjectAccessReviewSpec=lambda **kw: SimpleNamespace(**kw),
        V1ResourceAttributes=lambda **kw: SimpleNamespace(**kw),
        V1SelfSubjectReview=lambda: SimpleNamespace(),
    )


def _wire_admin(monkeypatch, rbac, *, core=None, self_review=None):
    """Happy admin path: load_clients stubbed (core + audit CM fake), _require_admin
    no-op, client -> fake. Returns the (possibly default) ``_AuditCore`` used, so
    tests can inspect what got written to the audit ConfigMap."""
    core = core if core is not None else _AuditCore(existing_log=[])
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "_require_admin", lambda: None)
    monkeypatch.setattr(sdk, "client", _client(rbac=rbac, self_review=self_review))
    return core


# =========================================================================== #
# name validation — happens BEFORE load_clients / _require_admin (no API touched)
# =========================================================================== #
def _tripwire(monkeypatch):
    """Fail loudly if grant_admin reaches load_clients / _require_admin / client."""
    monkeypatch.setattr(
        sdk, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not run")),
    )
    monkeypatch.setattr(
        sdk, "_require_admin",
        lambda: (_ for _ in ()).throw(AssertionError("_require_admin must not run")),
    )
    monkeypatch.setattr(
        sdk, "client",
        SimpleNamespace(RbacAuthorizationV1Api=lambda: (_ for _ in ()).throw(
            AssertionError("client must not be used"))),
    )


@pytest.mark.parametrize("bad", [
    "", "  ", "Alice", "a_b", "-alice", "alice-", "a.b", "al ice",
    "x" * 64,  # too long (>63)
])
def test_invalid_namespace_rejected_no_api(monkeypatch, bad):
    # no reason passed at all -- namespace format check fires first regardless.
    _tripwire(monkeypatch)
    with pytest.raises(WujiError):
        sdk.grant_admin(bad)


@pytest.mark.parametrize("bad", ["Foo", "a_b", "-x", "x-", "f o", "y" * 64])
def test_invalid_sa_rejected_no_api(monkeypatch, bad):
    _tripwire(monkeypatch)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", sa=bad)


def test_invalid_name_rejected_even_on_revoke(monkeypatch):
    # validation is unconditional, before the revoke branch too.
    _tripwire(monkeypatch)
    with pytest.raises(WujiError):
        sdk.grant_admin("Bad_Name", revoke=True)


# =========================================================================== #
# reason required — also happens before load_clients / _require_admin
# =========================================================================== #
@pytest.mark.parametrize("bad_reason", ["", "  ", None])
def test_reason_required_rejected_no_api(monkeypatch, bad_reason):
    _tripwire(monkeypatch)
    with pytest.raises(WujiError) as ei:
        sdk.grant_admin("alice", reason=bad_reason)
    assert "reason" in str(ei.value) or "理由" in str(ei.value)


def test_reason_required_rejected_on_revoke_too(monkeypatch):
    _tripwire(monkeypatch)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", reason="", revoke=True)


def test_reason_omitted_entirely_defaults_blank_and_rejected(monkeypatch):
    # reason has a "" default at the signature level (so malformed namespace/sa
    # still raise their OWN error first) but omitting it for an otherwise-valid
    # call must still be rejected, not silently treated as "no justification needed".
    _tripwire(monkeypatch)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice")


# =========================================================================== #
# admin gate (real _require_admin path)
# =========================================================================== #
def test_grant_requires_admin_denied(monkeypatch):
    # SSAR denied -> WujiError, before ANY RBAC write.
    rbac = _Rbac()
    core = _AuditCore(existing_log=[])
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "client", _client(rbac=rbac, auth=_Auth(allowed=False)))
    with pytest.raises(WujiError) as ei:
        sdk.grant_admin("alice", reason="onboarding")
    assert "管理员" in str(ei.value)
    assert rbac.names() == []  # nothing written


def test_revoke_requires_admin_denied(monkeypatch):
    # revoke is admin-gated too: denied -> WujiError, delete never called.
    rbac = _Rbac()
    core = _AuditCore(existing_log=[])
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "client", _client(rbac=rbac, auth=_Auth(allowed=False)))
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", reason="offboarding", revoke=True)
    assert "delete_cluster_role_binding" not in rbac.names()


def test_grant_require_admin_is_invoked(monkeypatch):
    called = {"n": 0}
    rbac = _Rbac()
    core = _AuditCore(existing_log=[])
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "_require_admin", lambda: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setattr(sdk, "client", _client(rbac=rbac))
    sdk.grant_admin("alice", reason="onboarding")
    assert called["n"] == 1


# =========================================================================== #
# grant — binds the built-in cluster-admin (only one tier, punchlist §10)
# =========================================================================== #
def test_grant_creates_binding_to_cluster_admin(monkeypatch):
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", reason="onboarding")

    # no custom ClusterRole is created/patched anymore -- only the binding.
    assert "create_cluster_role" not in rbac.names()
    assert "patch_cluster_role" not in rbac.names()

    # ClusterRoleBinding: name <ns>-admin, roleRef -> built-in cluster-admin.
    crb = rbac.created_crb
    assert crb["metadata"]["name"] == "alice-admin"
    assert crb["roleRef"]["kind"] == "ClusterRole"
    assert crb["roleRef"]["name"] == "cluster-admin"
    assert crb["roleRef"]["apiGroup"] == "rbac.authorization.k8s.io"

    # subject = the ns's SA (name == ns, in that ns).
    subj = crb["subjects"][0]
    assert subj["kind"] == "ServiceAccount"
    assert subj["name"] == "alice"
    assert subj["namespace"] == "alice"

    assert out == {
        "namespace": "alice", "sa": "alice",
        "role": "cluster-admin", "binding": "alice-admin", "audit_logged": True,
    }


def test_grant_call_order(monkeypatch):
    # just read existing binding, then create it -- no ClusterRole ops at all.
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    sdk.grant_admin("alice", reason="onboarding")
    assert rbac.names() == ["read_cluster_role_binding", "create_cluster_role_binding"]
    assert rbac.read_crb == "alice-admin"


# =========================================================================== #
# binding: read-before-create (roleRef immutable, no 409 swallow)
# =========================================================================== #
def test_grant_binding_exists_same_role_unchanged(monkeypatch):
    # existing binding already points at cluster-admin -> idempotent, no create.
    rbac = _Rbac(read_obj=_binding("cluster-admin"))
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", reason="re-confirm")
    assert out == {
        "namespace": "alice", "sa": "alice", "role": "cluster-admin",
        "binding": "alice-admin", "unchanged": True, "audit_logged": True,
    }
    assert "create_cluster_role_binding" not in rbac.names()


def test_grant_binding_exists_legacy_role_raises(monkeypatch):
    # a leftover binding to the old scoped-down "volcano-admin" role (from
    # before punchlist §10 collapsed the two tiers) must not be silently
    # upgraded -- error, revoke first (roleRef is immutable).
    rbac = _Rbac(read_obj=_binding("volcano-admin"))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError) as ei:
        sdk.grant_admin("alice", reason="upgrade legacy binding")
    assert "revoke" in str(ei.value) or "撤销" in str(ei.value)
    assert "create_cluster_role_binding" not in rbac.names()


def test_grant_binding_exists_role_ref_none_raises(monkeypatch):
    # defensive: existing binding with no role_ref -> cur_role None != role -> error.
    rbac = _Rbac(read_obj=SimpleNamespace(role_ref=None))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", reason="onboarding")
    assert "create_cluster_role_binding" not in rbac.names()


@pytest.mark.parametrize("status", [403, 500])
def test_grant_binding_read_other_exc_raises(monkeypatch, status):
    rbac = _Rbac(read_exc=ApiException(status=status))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", reason="onboarding")
    assert "create_cluster_role_binding" not in rbac.names()


@pytest.mark.parametrize("status", [403, 409, 500])
def test_grant_binding_create_exc_raises(monkeypatch, status):
    # read 404 -> create; create failing with ANY status (incl. unexpected 409) -> WujiError.
    rbac = _Rbac(create_crb_exc=ApiException(status=status))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", reason="onboarding")


# =========================================================================== #
# revoke -> delete the binding
# =========================================================================== #
def test_revoke_deletes_binding(monkeypatch):
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", reason="offboarding", revoke=True)
    assert rbac.deleted_crb == "alice-admin"
    assert out == {"namespace": "alice", "revoked": True, "audit_logged": True}
    # revoke touches nothing else.
    assert rbac.names() == ["delete_cluster_role_binding"]


def test_revoke_404_swallowed(monkeypatch):
    rbac = _Rbac(delete_crb_exc=ApiException(status=404))
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", reason="offboarding", revoke=True)  # 404 -> not an error
    assert out == {"namespace": "alice", "revoked": True, "audit_logged": True}


@pytest.mark.parametrize("status", [403, 409, 500])
def test_revoke_other_exc_raises(monkeypatch, status):
    rbac = _Rbac(delete_crb_exc=ApiException(status=status))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", reason="offboarding", revoke=True)


# =========================================================================== #
# sa override
# =========================================================================== #
def test_sa_override_subject_but_ns_binding_name(monkeypatch):
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", reason="onboarding", sa="foo")
    subj = rbac.created_crb["subjects"][0]
    assert subj["name"] == "foo"          # SA overridden
    assert subj["namespace"] == "alice"   # still in the person's ns
    assert out["sa"] == "foo"
    assert out["binding"] == "alice-admin"  # binding name stays ns-based
    assert rbac.created_crb["metadata"]["name"] == "alice-admin"


def test_sa_override_revoke_still_ns_binding(monkeypatch):
    # crb_name is built from namespace, not sa -> revoke deletes <ns>-admin.
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    sdk.grant_admin("alice", reason="offboarding", sa="foo", revoke=True)
    assert rbac.deleted_crb == "alice-admin"


# =========================================================================== #
# audit log — admin-grant-audit-log ConfigMap (wuji-system)
# =========================================================================== #
def test_audit_entry_created_when_cm_absent(monkeypatch):
    rbac = _Rbac()
    core = _AuditCore(existing_log=None)  # 404 on read -> create path
    _wire_admin(monkeypatch, rbac, core=core)
    sdk.grant_admin("alice", reason="onboarding new hire")

    assert core.created is not None
    assert core.created["metadata"]["name"] == sdk._ADMIN_GRANT_AUDIT_CM
    assert core.created["metadata"]["namespace"] == sdk._ADMIN_GRANT_AUDIT_NS
    log = json.loads(core.created["data"]["log"])
    assert len(log) == 1
    entry = log[0]
    assert entry["operator"] == "test-operator"
    assert entry["action"] == "grant"
    assert entry["namespace"] == "alice"
    assert entry["sa"] == "alice"
    assert entry["role"] == "cluster-admin"
    assert entry["reason"] == "onboarding new hire"
    assert "time" in entry


def test_audit_entry_appended_when_cm_exists(monkeypatch):
    rbac = _Rbac()
    prior = [{"time": "2026-01-01T00:00:00Z", "operator": "someone-else",
              "action": "grant", "namespace": "bob", "sa": "bob",
              "role": "cluster-admin", "reason": "past grant"}]
    core = _AuditCore(existing_log=prior)
    _wire_admin(monkeypatch, rbac, core=core)
    sdk.grant_admin("alice", reason="onboarding")

    assert core.patched is not None
    name, namespace, body = core.patched
    assert name == sdk._ADMIN_GRANT_AUDIT_CM
    assert namespace == sdk._ADMIN_GRANT_AUDIT_NS
    log = json.loads(body["data"]["log"])
    assert len(log) == 2
    assert log[0] == prior[0]          # prior entry preserved, untouched
    assert log[1]["namespace"] == "alice"
    assert core.created is None        # never hit the create branch


def test_audit_entry_action_revoke_vs_revoke_noop(monkeypatch):
    rbac = _Rbac()
    core = _AuditCore(existing_log=[])
    _wire_admin(monkeypatch, rbac, core=core)
    sdk.grant_admin("alice", reason="offboarding", revoke=True)
    log = json.loads(core.patched[2]["data"]["log"])
    assert log[-1]["action"] == "revoke"
    assert log[-1]["role"] is None

    rbac2 = _Rbac(delete_crb_exc=ApiException(status=404))
    core2 = _AuditCore(existing_log=[])
    _wire_admin(monkeypatch, rbac2, core=core2)
    sdk.grant_admin("bob", reason="already gone", revoke=True)
    log2 = json.loads(core2.patched[2]["data"]["log"])
    assert log2[-1]["action"] == "revoke_noop"


def test_audit_entry_action_grant_unchanged(monkeypatch):
    rbac = _Rbac(read_obj=_binding("cluster-admin"))
    core = _AuditCore(existing_log=[])
    _wire_admin(monkeypatch, rbac, core=core)
    sdk.grant_admin("alice", reason="re-confirm")
    log = json.loads(core.patched[2]["data"]["log"])
    assert log[-1]["action"] == "grant_unchanged"


def test_audit_write_failure_does_not_fail_grant(monkeypatch):
    # audit CM patch fails with a non-404 -> _log_admin_grant_audit swallows it,
    # grant_admin still succeeds (RBAC change already happened) but reports
    # audit_logged: False instead of raising.
    rbac = _Rbac()
    core = _AuditCore(existing_log=[], patch_exc=ApiException(status=500))
    _wire_admin(monkeypatch, rbac, core=core)
    out = sdk.grant_admin("alice", reason="onboarding")
    assert out["audit_logged"] is False
    assert rbac.created_crb is not None  # the actual grant still went through


def test_audit_write_failure_does_not_fail_revoke(monkeypatch):
    rbac = _Rbac()
    core = _AuditCore(existing_log=[], patch_exc=ApiException(status=500))
    _wire_admin(monkeypatch, rbac, core=core)
    out = sdk.grant_admin("alice", reason="offboarding", revoke=True)
    assert out["audit_logged"] is False
    assert rbac.deleted_crb == "alice-admin"  # the actual revoke still went through


def test_audit_create_failure_also_degrades_not_raises(monkeypatch):
    rbac = _Rbac()
    core = _AuditCore(existing_log=None, create_exc=ApiException(status=403))
    _wire_admin(monkeypatch, rbac, core=core)
    out = sdk.grant_admin("alice", reason="onboarding")
    assert out["audit_logged"] is False


def test_operator_identity_used_in_audit_entry(monkeypatch):
    rbac = _Rbac()
    core = _AuditCore(existing_log=[])
    _wire_admin(monkeypatch, rbac, core=core, self_review=_FakeSelfSubjectReview(username="203627766648865613"))
    sdk.grant_admin("alice", reason="onboarding")
    log = json.loads(core.patched[2]["data"]["log"])
    assert log[-1]["operator"] == "203627766648865613"


def test_operator_identity_unknown_on_review_failure(monkeypatch):
    rbac = _Rbac()
    core = _AuditCore(existing_log=[])
    _wire_admin(monkeypatch, rbac, core=core,
                self_review=_FakeSelfSubjectReview(exc=ApiException(status=403)))
    sdk.grant_admin("alice", reason="onboarding")
    log = json.loads(core.patched[2]["data"]["log"])
    assert log[-1]["operator"] == "unknown"


# =========================================================================== #
# CLI  (CliRunner + mocked sdk.grant_admin)
# =========================================================================== #
def _guard_no_kube(monkeypatch):
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )


def _capture_grant(monkeypatch, ret):
    captured = {}

    def fake(namespace, *, reason, sa=None, revoke=False):
        captured.update(namespace=namespace, reason=reason, sa=sa, revoke=revoke)
        return ret

    monkeypatch.setattr(cli.sdk, "grant_admin", fake)
    return captured


def test_cli_grant_default_prints_role(monkeypatch):
    _guard_no_kube(monkeypatch)
    cap = _capture_grant(monkeypatch, {
        "namespace": "alice", "sa": "alice",
        "role": "cluster-admin", "binding": "alice-admin", "audit_logged": True,
    })
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--reason", "onboarding"])
    assert result.exit_code == 0, result.output
    assert cap["namespace"] == "alice"
    assert cap["reason"] == "onboarding"
    assert cap["sa"] is None
    assert cap["revoke"] is False
    assert "设为管理员" in result.output
    assert "cluster-admin" in result.output


def test_cli_reason_required_by_typer(monkeypatch):
    # sdk.grant_admin must never even be reached -- typer itself rejects a
    # missing required option before dispatching to the command body.
    monkeypatch.setattr(
        cli.sdk, "grant_admin",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("grant_admin must not run")),
    )
    result = runner.invoke(cli.app, ["grant-admin", "alice"])
    assert result.exit_code != 0
    assert "reason" in result.output.lower() or "--reason" in result.output


def test_cli_no_cluster_admin_flag_exists(monkeypatch):
    # punchlist §10: the two-tier design was rejected, so --cluster-admin must
    # no longer be a recognized option at all.
    _guard_no_kube(monkeypatch)
    _capture_grant(monkeypatch, {"namespace": "alice", "revoked": True, "audit_logged": True})
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--reason", "x", "--cluster-admin"])
    assert result.exit_code != 0
    assert "no such option" in result.output.lower() or "unexpected" in result.output.lower()


def test_cli_revoke_prints_revoked(monkeypatch):
    _guard_no_kube(monkeypatch)
    cap = _capture_grant(monkeypatch, {"namespace": "alice", "revoked": True, "audit_logged": True})
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--reason", "offboarding", "--revoke"])
    assert result.exit_code == 0, result.output
    assert cap["revoke"] is True
    assert "已撤销" in result.output


def test_cli_sa_threaded(monkeypatch):
    _guard_no_kube(monkeypatch)
    cap = _capture_grant(monkeypatch, {
        "namespace": "alice", "sa": "foo",
        "role": "cluster-admin", "binding": "alice-admin", "audit_logged": True,
    })
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--reason", "onboarding", "--sa", "foo"])
    assert result.exit_code == 0, result.output
    assert cap["sa"] == "foo"
    assert "foo" in result.output  # printed SA in the confirmation line


def test_cli_audit_logged_false_prints_warning(monkeypatch):
    _guard_no_kube(monkeypatch)
    _capture_grant(monkeypatch, {
        "namespace": "alice", "sa": "alice",
        "role": "cluster-admin", "binding": "alice-admin", "audit_logged": False,
    })
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--reason", "onboarding"])
    assert result.exit_code == 0, result.output
    assert "审计记录写入失败" in result.output
    assert "设为管理员" in result.output  # the grant itself still reported as done


def test_cli_audit_logged_true_no_warning(monkeypatch):
    _guard_no_kube(monkeypatch)
    _capture_grant(monkeypatch, {
        "namespace": "alice", "sa": "alice",
        "role": "cluster-admin", "binding": "alice-admin", "audit_logged": True,
    })
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--reason", "onboarding"])
    assert result.exit_code == 0, result.output
    assert "审计记录写入失败" not in result.output


def test_cli_wuji_error_friendly_exit(monkeypatch):
    _guard_no_kube(monkeypatch)

    def boom(*a, **k):
        raise cli.WujiError("需要管理员 kubeconfig")

    monkeypatch.setattr(cli.sdk, "grant_admin", boom)
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--reason", "onboarding"])
    assert result.exit_code == 1
    assert "需要管理员" in result.output
    assert not isinstance(result.exception, cli.WujiError)  # no leaked traceback
