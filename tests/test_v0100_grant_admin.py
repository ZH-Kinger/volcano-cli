"""v0.10.0 tests: ``volcano grant-admin`` (admin makes someone a platform admin).

Fully mocked — no real cluster / kubeconfig. Two layers:

* SDK ``sdk.grant_admin(namespace, *, sa=None, revoke=False, cluster_admin=False)``:
  ``load_clients`` → ``_require_admin`` gate → ``RbacAuthorizationV1Api``. Ensures
  the shared ``volcano-admin`` ClusterRole (create-or-patch to ``_VOLCANO_ADMIN_RULES``,
  skipped for ``--cluster-admin``), then binds ``<ns>``'s SA via ClusterRoleBinding
  ``<ns>-admin`` (409 → idempotent). ``--revoke`` deletes the binding (404 swallowed).
* CLI ``grant-admin`` (grant_admin_cmd): threads namespace/--sa/--cluster-admin/--revoke
  into ``sdk.grant_admin`` and prints friendly lines; a ``WujiError`` → non-zero exit.

Mocking approach mirrors tests/test_v090_sdk_quota_queue.py: monkeypatch
``sdk.load_clients`` / ``sdk._require_admin`` and swap the kubernetes ``client``
module object on ``volcano.sdk`` so ``client.RbacAuthorizationV1Api()`` (and, for the
real-gate test, ``client.AuthorizationV1Api()`` + SSAR builders) return our fakes.
Only tests/ is touched.
"""

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


def _binding(role):
    """A fake existing ClusterRoleBinding whose roleRef points at ``role``."""
    return SimpleNamespace(role_ref=SimpleNamespace(name=role))


class _Rbac:
    """Records RBAC calls; each ``*_exc`` raises from the matching call.

    ``read_obj`` / ``read_exc`` drive ``read_cluster_role_binding``: an object to
    return (existing binding), an exception to raise, else a default 404 (absent).
    """

    def __init__(self, *, create_cr_exc=None, patch_cr_exc=None,
                 create_crb_exc=None, delete_crb_exc=None,
                 read_obj=None, read_exc=None):
        self.create_cr_exc = create_cr_exc
        self.patch_cr_exc = patch_cr_exc
        self.create_crb_exc = create_crb_exc
        self.delete_crb_exc = delete_crb_exc
        self.read_obj = read_obj
        self.read_exc = read_exc
        self.calls = []
        self.created_cr = None
        self.patched_cr = None          # (name, body)
        self.created_crb = None
        self.deleted_crb = None         # name
        self.read_crb = None            # name

    def names(self):
        return [c[0] for c in self.calls]

    def create_cluster_role(self, body):
        self.calls.append(("create_cluster_role", body))
        self.created_cr = body
        if self.create_cr_exc:
            raise self.create_cr_exc

    def patch_cluster_role(self, name, body):
        self.calls.append(("patch_cluster_role", name, body))
        self.patched_cr = (name, body)
        if self.patch_cr_exc:
            raise self.patch_cr_exc

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


def _client(*, rbac=None, auth=None):
    """Stand-in for the ``kubernetes.client`` module object used inside sdk."""
    return SimpleNamespace(
        RbacAuthorizationV1Api=lambda: rbac,
        AuthorizationV1Api=lambda: auth,
        V1SelfSubjectAccessReview=lambda **kw: SimpleNamespace(**kw),
        V1SelfSubjectAccessReviewSpec=lambda **kw: SimpleNamespace(**kw),
        V1ResourceAttributes=lambda **kw: SimpleNamespace(**kw),
    )


def _wire_admin(monkeypatch, rbac):
    """Happy admin path: load_clients stubbed, _require_admin no-op, client → fake."""
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, None))
    monkeypatch.setattr(sdk, "_require_admin", lambda: None)
    monkeypatch.setattr(sdk, "client", _client(rbac=rbac))


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
# admin gate (real _require_admin path)
# =========================================================================== #
def test_grant_requires_admin_denied(monkeypatch):
    # SSAR denied -> WujiError, before ANY RBAC write.
    rbac = _Rbac()
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, None))
    monkeypatch.setattr(sdk, "client", _client(rbac=rbac, auth=_Auth(allowed=False)))
    with pytest.raises(WujiError) as ei:
        sdk.grant_admin("alice")
    assert "管理员" in str(ei.value)
    assert rbac.names() == []  # nothing written


def test_revoke_requires_admin_denied(monkeypatch):
    # revoke is admin-gated too: denied -> WujiError, delete never called.
    rbac = _Rbac()
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, None))
    monkeypatch.setattr(sdk, "client", _client(rbac=rbac, auth=_Auth(allowed=False)))
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", revoke=True)
    assert "delete_cluster_role_binding" not in rbac.names()


def test_grant_require_admin_is_invoked(monkeypatch):
    called = {"n": 0}
    rbac = _Rbac()
    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, None))
    monkeypatch.setattr(sdk, "_require_admin", lambda: called.__setitem__("n", called["n"] + 1))
    monkeypatch.setattr(sdk, "client", _client(rbac=rbac))
    sdk.grant_admin("alice")
    assert called["n"] == 1


# =========================================================================== #
# grant — default volcano-admin
# =========================================================================== #
def test_grant_default_creates_role_and_binding(monkeypatch):
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice")

    # ClusterRole created with the converged rules; no patch on a fresh create.
    assert rbac.created_cr["metadata"]["name"] == "volcano-admin"
    assert rbac.created_cr["rules"] == sdk._VOLCANO_ADMIN_RULES
    assert rbac.created_cr["rules"] is sdk._VOLCANO_ADMIN_RULES
    assert "patch_cluster_role" not in rbac.names()

    # ClusterRoleBinding: name <ns>-admin, roleRef -> volcano-admin ClusterRole.
    crb = rbac.created_crb
    assert crb["metadata"]["name"] == "alice-admin"
    assert crb["roleRef"]["kind"] == "ClusterRole"
    assert crb["roleRef"]["name"] == "volcano-admin"
    assert crb["roleRef"]["apiGroup"] == "rbac.authorization.k8s.io"

    # subject = the ns's SA (name == ns, in that ns).
    subj = crb["subjects"][0]
    assert subj["kind"] == "ServiceAccount"
    assert subj["name"] == "alice"
    assert subj["namespace"] == "alice"

    assert out == {
        "namespace": "alice", "sa": "alice",
        "role": "volcano-admin", "binding": "alice-admin",
    }


def test_grant_default_call_order(monkeypatch):
    # role ensured, then read existing binding, then create it.
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    sdk.grant_admin("alice")
    assert rbac.names() == [
        "create_cluster_role", "read_cluster_role_binding", "create_cluster_role_binding",
    ]
    assert rbac.read_crb == "alice-admin"


# =========================================================================== #
# ClusterRole already exists -> reconcile via patch
# =========================================================================== #
def test_grant_role_exists_409_patches(monkeypatch):
    rbac = _Rbac(create_cr_exc=ApiException(status=409))
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice")

    # 409 -> patch once with the rules; binding still created afterwards.
    assert rbac.patched_cr is not None
    assert rbac.patched_cr[0] == "volcano-admin"
    assert rbac.patched_cr[1] == {"rules": sdk._VOLCANO_ADMIN_RULES}
    assert rbac.created_crb["metadata"]["name"] == "alice-admin"
    assert out["role"] == "volcano-admin"
    assert rbac.names() == [
        "create_cluster_role", "patch_cluster_role",
        "read_cluster_role_binding", "create_cluster_role_binding",
    ]


def test_grant_role_patch_other_exc_raises(monkeypatch):
    # create 409 -> patch, but patch itself fails -> WujiError.
    rbac = _Rbac(create_cr_exc=ApiException(status=409),
                 patch_cr_exc=ApiException(status=500))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice")
    assert "create_cluster_role_binding" not in rbac.names()  # aborted before binding


def test_grant_role_create_other_exc_raises(monkeypatch):
    # create_cluster_role non-409 -> WujiError, no patch, no binding.
    rbac = _Rbac(create_cr_exc=ApiException(status=403))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice")
    assert "patch_cluster_role" not in rbac.names()
    assert "create_cluster_role_binding" not in rbac.names()


# =========================================================================== #
# binding: read-before-create (roleRef immutable, no 409 swallow)
# =========================================================================== #
def test_grant_binding_exists_same_role_unchanged(monkeypatch):
    # existing binding already points at volcano-admin -> idempotent, no create.
    rbac = _Rbac(read_obj=_binding("volcano-admin"))
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice")
    assert out == {
        "namespace": "alice", "sa": "alice",
        "role": "volcano-admin", "binding": "alice-admin", "unchanged": True,
    }
    assert "create_cluster_role_binding" not in rbac.names()


def test_grant_binding_exists_different_role_raises(monkeypatch):
    # already bound to cluster-admin, now asked for volcano-admin -> error (revoke first).
    rbac = _Rbac(read_obj=_binding("cluster-admin"))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError) as ei:
        sdk.grant_admin("alice")  # default role = volcano-admin
    assert "revoke" in str(ei.value) or "撤销" in str(ei.value)
    assert "create_cluster_role_binding" not in rbac.names()


def test_grant_binding_exists_role_ref_none_raises(monkeypatch):
    # defensive: existing binding with no role_ref -> cur_role None != role -> error.
    rbac = _Rbac(read_obj=SimpleNamespace(role_ref=None))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice")
    assert "create_cluster_role_binding" not in rbac.names()


@pytest.mark.parametrize("status", [403, 500])
def test_grant_binding_read_other_exc_raises(monkeypatch, status):
    rbac = _Rbac(read_exc=ApiException(status=status))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice")
    assert "create_cluster_role_binding" not in rbac.names()


@pytest.mark.parametrize("status", [403, 409, 500])
def test_grant_binding_create_exc_raises(monkeypatch, status):
    # read 404 -> create; create failing with ANY status (incl. unexpected 409) -> WujiError.
    rbac = _Rbac(create_crb_exc=ApiException(status=status))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice")


# =========================================================================== #
# cluster_admin=True -> bind built-in cluster-admin, skip the ClusterRole
# =========================================================================== #
def test_grant_cluster_admin_skips_role(monkeypatch):
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", cluster_admin=True)

    assert "create_cluster_role" not in rbac.names()
    assert "patch_cluster_role" not in rbac.names()
    assert rbac.created_cr is None
    assert rbac.created_crb["roleRef"]["name"] == "cluster-admin"
    assert rbac.created_crb["roleRef"]["kind"] == "ClusterRole"
    assert out["role"] == "cluster-admin"
    assert out["binding"] == "alice-admin"
    # no ClusterRole ops; just read (absent) then create the binding.
    assert rbac.names() == ["read_cluster_role_binding", "create_cluster_role_binding"]


def test_grant_cluster_admin_existing_same_role_unchanged(monkeypatch):
    rbac = _Rbac(read_obj=_binding("cluster-admin"))
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", cluster_admin=True)
    assert out["role"] == "cluster-admin"
    assert out["unchanged"] is True
    assert "create_cluster_role_binding" not in rbac.names()


def test_grant_cluster_admin_over_volcano_admin_raises(monkeypatch):
    # already bound volcano-admin, now asked for cluster-admin -> revoke first.
    rbac = _Rbac(read_obj=_binding("volcano-admin"))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", cluster_admin=True)
    assert "create_cluster_role_binding" not in rbac.names()


# =========================================================================== #
# revoke -> delete the binding
# =========================================================================== #
def test_revoke_deletes_binding(monkeypatch):
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", revoke=True)
    assert rbac.deleted_crb == "alice-admin"
    assert out == {"namespace": "alice", "revoked": True}
    # revoke touches nothing else.
    assert rbac.names() == ["delete_cluster_role_binding"]


def test_revoke_404_swallowed(monkeypatch):
    rbac = _Rbac(delete_crb_exc=ApiException(status=404))
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", revoke=True)  # 404 -> not an error
    assert out == {"namespace": "alice", "revoked": True}


@pytest.mark.parametrize("status", [403, 409, 500])
def test_revoke_other_exc_raises(monkeypatch, status):
    rbac = _Rbac(delete_crb_exc=ApiException(status=status))
    _wire_admin(monkeypatch, rbac)
    with pytest.raises(WujiError):
        sdk.grant_admin("alice", revoke=True)


def test_revoke_ignores_cluster_admin_flag(monkeypatch):
    # revoke short-circuits: cluster_admin is irrelevant, binding name is ns-based.
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", revoke=True, cluster_admin=True)
    assert rbac.deleted_crb == "alice-admin"
    assert out == {"namespace": "alice", "revoked": True}


# =========================================================================== #
# sa override
# =========================================================================== #
def test_sa_override_subject_but_ns_binding_name(monkeypatch):
    rbac = _Rbac()
    _wire_admin(monkeypatch, rbac)
    out = sdk.grant_admin("alice", sa="foo")
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
    sdk.grant_admin("alice", sa="foo", revoke=True)
    assert rbac.deleted_crb == "alice-admin"


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

    def fake(namespace, *, sa=None, revoke=False, cluster_admin=False):
        captured.update(namespace=namespace, sa=sa, revoke=revoke,
                        cluster_admin=cluster_admin)
        return ret

    monkeypatch.setattr(cli.sdk, "grant_admin", fake)
    return captured


def test_cli_grant_default_prints_role(monkeypatch):
    _guard_no_kube(monkeypatch)
    cap = _capture_grant(monkeypatch, {
        "namespace": "alice", "sa": "alice",
        "role": "volcano-admin", "binding": "alice-admin",
    })
    result = runner.invoke(cli.app, ["grant-admin", "alice"])
    assert result.exit_code == 0, result.output
    assert cap["namespace"] == "alice"
    assert cap["sa"] is None
    assert cap["revoke"] is False and cap["cluster_admin"] is False
    assert "设为管理员" in result.output
    assert "volcano-admin" in result.output


def test_cli_grant_cluster_admin(monkeypatch):
    _guard_no_kube(monkeypatch)
    cap = _capture_grant(monkeypatch, {
        "namespace": "alice", "sa": "alice",
        "role": "cluster-admin", "binding": "alice-admin",
    })
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--cluster-admin"])
    assert result.exit_code == 0, result.output
    assert cap["cluster_admin"] is True
    assert "cluster-admin" in result.output


def test_cli_revoke_prints_revoked(monkeypatch):
    _guard_no_kube(monkeypatch)
    cap = _capture_grant(monkeypatch, {"namespace": "alice", "revoked": True})
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--revoke"])
    assert result.exit_code == 0, result.output
    assert cap["revoke"] is True
    assert "已撤销" in result.output


def test_cli_sa_threaded(monkeypatch):
    _guard_no_kube(monkeypatch)
    cap = _capture_grant(monkeypatch, {
        "namespace": "alice", "sa": "foo",
        "role": "volcano-admin", "binding": "alice-admin",
    })
    result = runner.invoke(cli.app, ["grant-admin", "alice", "--sa", "foo"])
    assert result.exit_code == 0, result.output
    assert cap["sa"] == "foo"
    assert "foo" in result.output  # printed SA in the confirmation line


def test_cli_wuji_error_friendly_exit(monkeypatch):
    _guard_no_kube(monkeypatch)

    def boom(*a, **k):
        raise cli.WujiError("需要管理员 kubeconfig")

    monkeypatch.setattr(cli.sdk, "grant_admin", boom)
    result = runner.invoke(cli.app, ["grant-admin", "alice"])
    assert result.exit_code == 1
    assert "需要管理员" in result.output
    assert not isinstance(result.exception, cli.WujiError)  # no leaked traceback
