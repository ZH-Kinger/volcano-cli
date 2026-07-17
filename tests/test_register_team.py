"""Tests for the admin-only ``volcano register`` team-onboarding surface.

Covers ``volcano.sdk.register_team`` plus its helpers (``_ignore_conflict``,
``_require_admin``, ``_nas_mkdir``, ``_TEAM_NAME_RE``) and the ``register`` CLI
command. Everything is fully mocked — no real cluster, no kubeconfig, no waiting
(``time.sleep`` is stubbed). Only this tests/ tree is touched.

The kubernetes ``client`` module object is swapped on ``volcano.sdk`` so the
``client.CoreV1Api()`` / ``RbacAuthorizationV1Api()`` / ``AuthorizationV1Api()``
instantiations inside ``register_team`` return our fakes.
"""

import base64
from types import SimpleNamespace

import pytest
import yaml
from kubernetes.client.exceptions import ApiException

import volcano.cli as cli
import volcano.sdk as sdk
from volcano.kube import DEFAULT_APISERVER, NAS_BASE, NAS_SERVER, WujiError


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _secret(token=b"secret-token-xyz", ca_b64="Q0EtUEVN"):
    """A fake CoreV1 Secret whose .data mimics the k8s wire format.

    ``token`` is base64-encoded (register_team b64-decodes it); ``ca.crt`` is
    passed through verbatim (already base64, kubeconfig wants it that way).
    """
    return SimpleNamespace(
        data={"token": base64.b64encode(token).decode(), "ca.crt": ca_b64}
    )


class FakeCore:
    """Records every create/read/delete call register_team + _nas_mkdir make."""

    def __init__(self, pod_phase="Succeeded", pod_log="MKDIR-OK\n", secret_reads=None):
        self.pod_phase = pod_phase
        self.pod_log = pod_log
        # sequence of SimpleNamespace(data=...) returned by read_namespaced_secret;
        # last item repeats once exhausted. Default: token present immediately.
        self.secret_reads = list(secret_reads) if secret_reads is not None else [_secret()]
        self.calls = []  # (method, args, kwargs)

    def _rec(self, _method, *a, **k):
        self.calls.append((_method, a, k))

    def names(self):
        return [c[0] for c in self.calls]

    # --- _nas_mkdir --------------------------------------------------------- #
    def delete_namespaced_pod(self, *a, **k):
        self._rec("delete_namespaced_pod", *a, **k)

    def create_namespaced_pod(self, *a, **k):
        self._rec("create_namespaced_pod", *a, **k)

    def read_namespaced_pod(self, *a, **k):
        self._rec("read_namespaced_pod", *a, **k)
        return SimpleNamespace(status=SimpleNamespace(phase=self.pod_phase))

    def read_namespaced_pod_log(self, *a, **k):
        self._rec("read_namespaced_pod_log", *a, **k)
        return self.pod_log

    # --- register_team core objects ---------------------------------------- #
    def create_namespace(self, *a, **k):
        self._rec("create_namespace", *a, **k)

    def create_persistent_volume(self, *a, **k):
        self._rec("create_persistent_volume", *a, **k)

    def create_namespaced_persistent_volume_claim(self, *a, **k):
        self._rec("create_namespaced_persistent_volume_claim", *a, **k)

    def create_namespaced_service_account(self, *a, **k):
        self._rec("create_namespaced_service_account", *a, **k)

    def create_namespaced_secret(self, *a, **k):
        self._rec("create_namespaced_secret", *a, **k)

    def read_namespaced_secret(self, *a, **k):
        self._rec("read_namespaced_secret", *a, **k)
        item = self.secret_reads.pop(0) if len(self.secret_reads) > 1 else self.secret_reads[0]
        return item

    # --- ACR tripwire: current code must never touch these ------------------ #
    def patch_namespaced_service_account(self, *a, **k):  # pragma: no cover
        self._rec("patch_namespaced_service_account", *a, **k)


class FakeRbac:
    def __init__(self):
        self.calls = []

    def _rec(self, _method, *a, **k):
        self.calls.append((_method, a, k))

    def create_namespaced_role(self, *a, **k):
        self._rec("create_namespaced_role", *a, **k)

    def create_namespaced_role_binding(self, *a, **k):
        self._rec("create_namespaced_role_binding", *a, **k)

    def create_cluster_role(self, *a, **k):
        self._rec("create_cluster_role", *a, **k)

    def create_cluster_role_binding(self, *a, **k):
        self._rec("create_cluster_role_binding", *a, **k)


class FakeAuth:
    """SelfSubjectAccessReview responder. ``allowed`` is a bool (all checks) or a
    list consumed in order; ``exc`` raises instead."""

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


def _make_client(core=None, rbac=None, auth=None):
    """Assemble a stand-in for the ``kubernetes.client`` module object."""
    return SimpleNamespace(
        CoreV1Api=lambda: core,
        RbacAuthorizationV1Api=lambda: rbac,
        AuthorizationV1Api=lambda: auth,
        V1SelfSubjectAccessReview=lambda **kw: SimpleNamespace(**kw),
        V1SelfSubjectAccessReviewSpec=lambda **kw: SimpleNamespace(**kw),
        V1ResourceAttributes=lambda **kw: SimpleNamespace(**kw),
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(sdk.time, "sleep", lambda *_a, **_k: None)


def _wire(monkeypatch, core, rbac, auth):
    """Patch sdk.load_clients + sdk.client for a full register_team run."""
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    monkeypatch.setattr(sdk, "client", _make_client(core, rbac, auth))


# =========================================================================== #
# 1) name validation — rejects before ANY k8s call
# =========================================================================== #
BAD_NAMES = [
    pytest.param("TeamA", id="uppercase"),
    pytest.param("team_a", id="underscore"),
    pytest.param("-team", id="leading-dash"),
    pytest.param("team-", id="trailing-dash"),
    pytest.param("a" * 64, id="too-long"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace-only"),
    pytest.param("te am", id="space"),
    pytest.param("tEAM/x", id="slash"),
]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_register_invalid_name_raises_and_no_api_touched(monkeypatch, name):
    hit = []
    monkeypatch.setattr(sdk, "load_clients", lambda: hit.append("load_clients"))
    # any client attribute access = the tripwire fired
    monkeypatch.setattr(
        sdk, "client",
        SimpleNamespace(
            CoreV1Api=lambda: hit.append("core") or None,
            RbacAuthorizationV1Api=lambda: hit.append("rbac") or None,
            AuthorizationV1Api=lambda: hit.append("auth") or None,
        ),
    )
    with pytest.raises(WujiError):
        sdk.register_team(name)
    assert hit == []  # validation must precede any cluster work


def test_register_63_char_name_is_allowed(monkeypatch):
    core, rbac, auth = FakeCore(), FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    out = sdk.register_team("a" * 63)
    assert out["team"] == "a" * 63


def test_register_name_is_stripped(monkeypatch):
    core, rbac, auth = FakeCore(), FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    out = sdk.register_team("  teamx  ")
    assert out["team"] == "teamx" and out["namespace"] == "teamx"


# =========================================================================== #
# 2) dry_run — plan only, creates nothing
# =========================================================================== #
def test_dry_run_returns_plan_and_touches_nothing(monkeypatch):
    hit = []
    monkeypatch.setattr(sdk, "load_clients", lambda: hit.append("load_clients"))
    monkeypatch.setattr(
        sdk, "client",
        SimpleNamespace(CoreV1Api=lambda: hit.append("core") or None),
    )
    out = sdk.register_team("teamx", dry_run=True)
    assert hit == []
    assert out["kubeconfig"] == ""
    assert out["created"] == []
    assert isinstance(out["plan"], list) and out["plan"]
    assert out["namespace"] == "teamx"


def test_dry_run_plan_reflects_custom_args(monkeypatch):
    monkeypatch.setattr(sdk, "load_clients", lambda: pytest.fail("no api in dry-run"))
    out = sdk.register_team(
        "teamx", apiserver="https://1.2.3.4:6443", storage="5Ti", dry_run=True
    )
    joined = "\n".join(out["plan"])
    assert "https://1.2.3.4:6443" in joined
    assert "5Ti" in joined


# =========================================================================== #
# 3) admin gate — _require_admin
# =========================================================================== #
def test_require_admin_all_allowed_passes(monkeypatch):
    auth = FakeAuth(allowed=True)
    monkeypatch.setattr(sdk, "client", _make_client(auth=auth))
    sdk._require_admin()  # no raise
    assert len(auth.reviews) == 2  # namespaces + clusterrolebindings both checked


def test_require_admin_denied_raises(monkeypatch):
    auth = FakeAuth(allowed=False)
    monkeypatch.setattr(sdk, "client", _make_client(auth=auth))
    with pytest.raises(WujiError) as ei:
        sdk._require_admin()
    assert "管理员" in str(ei.value)


def test_require_admin_second_check_denied_raises(monkeypatch):
    # can create namespaces but NOT clusterrolebindings → still blocked
    auth = FakeAuth(allowed=[True, False])
    monkeypatch.setattr(sdk, "client", _make_client(auth=auth))
    with pytest.raises(WujiError):
        sdk._require_admin()


def test_require_admin_api_error_raises_wuji(monkeypatch):
    auth = FakeAuth(exc=ApiException(status=500))
    monkeypatch.setattr(sdk, "client", _make_client(auth=auth))
    with pytest.raises(WujiError):
        sdk._require_admin()


def test_register_blocked_when_not_admin(monkeypatch):
    core, rbac, auth = FakeCore(), FakeRbac(), FakeAuth(allowed=False)
    _wire(monkeypatch, core, rbac, auth)
    with pytest.raises(WujiError):
        sdk.register_team("teamx")
    # gate fires before any namespace/pod creation
    assert "create_namespace" not in core.names()
    assert "create_namespaced_pod" not in core.names()


# =========================================================================== #
# 4) idempotency — _ignore_conflict
# =========================================================================== #
def test_ignore_conflict_success_returns_true():
    assert sdk._ignore_conflict(lambda: None) is True


def test_ignore_conflict_409_swallowed_returns_false():
    def boom():
        raise ApiException(status=409)

    assert sdk._ignore_conflict(boom) is False


@pytest.mark.parametrize("status", [403, 404, 422, 500])
def test_ignore_conflict_other_status_raises_wuji(status):
    def boom():
        raise ApiException(status=status)

    with pytest.raises(WujiError):
        sdk._ignore_conflict(boom)


def test_ignore_conflict_forwards_args():
    seen = {}

    def fn(a, b, kw=None):
        seen.update(a=a, b=b, kw=kw)

    assert sdk._ignore_conflict(fn, 1, 2, kw=3) is True
    assert seen == {"a": 1, "b": 2, "kw": 3}


# =========================================================================== #
# 5) token wait + 6) kubeconfig assembly
# =========================================================================== #
def test_token_appears_after_polling(monkeypatch):
    # first read: no token; second read: token present → must succeed
    core = FakeCore(secret_reads=[SimpleNamespace(data={}), _secret(b"tok-late")])
    rbac, auth = FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    out = sdk.register_team("teamx")
    doc = yaml.safe_load(out["kubeconfig"])
    assert doc["users"][0]["user"]["token"] == "tok-late"
    # polled at least twice
    assert core.names().count("read_namespaced_secret") >= 2


def test_token_never_injected_raises(monkeypatch):
    core = FakeCore(secret_reads=[SimpleNamespace(data={})])  # never any token
    rbac, auth = FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    with pytest.raises(WujiError) as ei:
        sdk.register_team("teamx")
    assert "token" in str(ei.value)


def test_token_secret_with_none_data_raises(monkeypatch):
    core = FakeCore(secret_reads=[SimpleNamespace(data=None)])
    rbac, auth = FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    with pytest.raises(WujiError):
        sdk.register_team("teamx")


def test_token_secret_read_api_error_then_success(monkeypatch):
    # transient ApiException on read must not abort the poll loop
    calls = {"n": 0}
    good = _secret(b"tok-ok")

    def read(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ApiException(status=500)
        return good

    core = FakeCore()
    core.read_namespaced_secret = read
    rbac, auth = FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    out = sdk.register_team("teamx")
    assert yaml.safe_load(out["kubeconfig"])["users"][0]["user"]["token"] == "tok-ok"


def test_kubeconfig_fully_assembled(monkeypatch):
    core = FakeCore(secret_reads=[_secret(b"my-token", ca_b64="Q0EtREFUQQ==")])
    rbac, auth = FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    out = sdk.register_team("teamx", apiserver="https://9.9.9.9:6443")
    doc = yaml.safe_load(out["kubeconfig"])
    assert doc["clusters"][0]["cluster"]["server"] == "https://9.9.9.9:6443"
    assert doc["clusters"][0]["cluster"]["certificate-authority-data"] == "Q0EtREFUQQ=="
    assert doc["users"][0]["user"]["token"] == "my-token"
    assert doc["contexts"][0]["context"]["namespace"] == "teamx"
    assert doc["current-context"] == "teamx"
    assert out["nas_pvc"] == "teamx-nas"
    assert out["apiserver"] == "https://9.9.9.9:6443"


def test_default_apiserver_used_when_unspecified(monkeypatch):
    core, rbac, auth = FakeCore(), FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    out = sdk.register_team("teamx")
    doc = yaml.safe_load(out["kubeconfig"])
    assert doc["clusters"][0]["cluster"]["server"] == DEFAULT_APISERVER


def test_created_list_reports_new_objects(monkeypatch):
    core, rbac, auth = FakeCore(), FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    out = sdk.register_team("teamx")
    # everything created fresh (no conflicts) → all listed
    assert f"namespace/teamx" in out["created"]
    assert "pv/teamx-nas-pv" in out["created"]
    assert "pvc/teamx-nas" in out["created"]
    assert "sa/teamx" in out["created"]
    assert "role/teamx-volcano" in out["created"]
    assert "rolebinding/teamx-volcano" in out["created"]
    assert "clusterrolebinding/teamx-volcano-viewer" in out["created"]
    assert "secret/teamx-token" in out["created"]


def test_existing_objects_not_reported_created(monkeypatch):
    # every create returns 409 → nothing "created", but kubeconfig still produced
    core, rbac, auth = FakeCore(), FakeRbac(), FakeAuth()

    def conflict(*a, **k):
        raise ApiException(status=409)

    for m in (
        "create_namespace", "create_persistent_volume",
        "create_namespaced_persistent_volume_claim",
        "create_namespaced_service_account", "create_namespaced_secret",
    ):
        setattr(core, m, conflict)
    for m in (
        "create_namespaced_role", "create_namespaced_role_binding",
        "create_cluster_role", "create_cluster_role_binding",
    ):
        setattr(rbac, m, conflict)
    _wire(monkeypatch, core, rbac, auth)
    out = sdk.register_team("teamx")
    assert out["created"] == []
    assert out["kubeconfig"]  # still assembled from the (pre-existing) token secret


# =========================================================================== #
# 7) never touches ACR
# =========================================================================== #
def test_register_never_touches_acr(monkeypatch):
    core, rbac, auth = FakeCore(), FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    sdk.register_team("teamx")
    # the only Secret created is the SA token — never a docker-registry secret
    secret_calls = [c for c in core.calls if c[0] == "create_namespaced_secret"]
    assert len(secret_calls) == 1
    body = secret_calls[0][1][1]  # positional (namespace, body)
    assert body["type"] == "kubernetes.io/service-account-token"
    assert body["metadata"]["name"] == "teamx-token"
    assert "dockerconfigjson" not in str(body).lower()
    # default SA imagePullSecrets never patched
    assert "patch_namespaced_service_account" not in core.names()


# =========================================================================== #
# 8) _nas_mkdir
# =========================================================================== #
def test_nas_mkdir_success_deletes_pod(monkeypatch):
    core = FakeCore(pod_phase="Succeeded", pod_log="line\nMKDIR-OK\n")
    sdk._nas_mkdir(core, "teamx", NAS_SERVER, NAS_BASE)
    names = core.names()
    assert "create_namespaced_pod" in names
    # deleted at least twice (best-effort pre-clean + final cleanup)
    assert names.count("delete_namespaced_pod") >= 2


def test_nas_mkdir_phase_failed_raises(monkeypatch):
    core = FakeCore(pod_phase="Failed", pod_log="boom")
    with pytest.raises(WujiError):
        sdk._nas_mkdir(core, "teamx", NAS_SERVER, NAS_BASE)
    # still cleans up the pod
    assert "delete_namespaced_pod" in core.names()


def test_nas_mkdir_succeeded_but_no_marker_raises(monkeypatch):
    core = FakeCore(pod_phase="Succeeded", pod_log="did nothing")
    with pytest.raises(WujiError):
        sdk._nas_mkdir(core, "teamx", NAS_SERVER, NAS_BASE)
    assert "delete_namespaced_pod" in core.names()


def test_nas_mkdir_create_pod_api_error_raises(monkeypatch):
    core = FakeCore()

    def boom(*a, **k):
        raise ApiException(status=403)

    core.create_namespaced_pod = boom
    with pytest.raises(WujiError):
        sdk._nas_mkdir(core, "teamx", NAS_SERVER, NAS_BASE)


def test_nas_mkdir_body_uses_nas_coords(monkeypatch):
    core = FakeCore()
    sdk._nas_mkdir(core, "teamx", "10.1.2.3", "/base/path")
    create = [c for c in core.calls if c[0] == "create_namespaced_pod"][0]
    body = create[2]["body"]
    vol = body["spec"]["volumes"][0]["nfs"]
    assert vol["server"] == "10.1.2.3"
    assert vol["path"] == "/base/path"
    # mkdir targets the team subdir
    assert "teamx" in body["spec"]["containers"][0]["command"][-1]


def test_register_calls_nas_mkdir(monkeypatch):
    # register_team must attempt the NAS dir creation between ns and PV
    core, rbac, auth = FakeCore(), FakeRbac(), FakeAuth()
    _wire(monkeypatch, core, rbac, auth)
    sdk.register_team("teamx")
    assert "create_namespaced_pod" in core.names()


# =========================================================================== #
# CLI — register command (sdk.register_team mocked)
# =========================================================================== #
from typer.testing import CliRunner  # noqa: E402

runner = CliRunner()


def _fake_register(**ret):
    def _f(team, **kwargs):
        base = {
            "team": team, "namespace": team, "kubeconfig": "KUBECONFIG-DATA\n",
            "apiserver": DEFAULT_APISERVER, "nas_pvc": f"{team}-nas",
            "created": ["namespace/" + team], "plan": [f"namespace/{team}", "pv/..."],
        }
        base.update(ret)
        return base

    return _f


def test_cli_dry_run_prints_plan_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sdk, "register_team", _fake_register())
    target = tmp_path / "teamx.kubeconfig"
    result = runner.invoke(
        cli.app, ["register", "teamx", "--dry-run", "--out", str(target)]
    )
    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "namespace/teamx" in result.output
    assert not target.exists()


def test_cli_dry_run_passes_flag_to_sdk(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli.sdk, "register_team",
        lambda team, **kw: seen.update(kw) or _fake_register()(team, **kw),
    )
    result = runner.invoke(cli.app, ["register", "teamx", "--dry-run"])
    assert result.exit_code == 0
    assert seen.get("dry_run") is True


def test_cli_writes_kubeconfig_to_out(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sdk, "register_team", _fake_register())
    target = tmp_path / "sub" / "teamx.kubeconfig"
    result = runner.invoke(cli.app, ["register", "teamx", "--out", str(target)])
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "KUBECONFIG-DATA\n"
    assert "已开通" in result.output


def test_cli_refuses_overwrite_without_force(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sdk, "register_team", _fake_register())
    target = tmp_path / "teamx.kubeconfig"
    target.write_text("OLD-CONTENT", encoding="utf-8")
    result = runner.invoke(cli.app, ["register", "teamx", "--out", str(target)])
    assert result.exit_code != 0
    assert "已存在" in result.output
    assert target.read_text(encoding="utf-8") == "OLD-CONTENT"  # untouched


def test_cli_force_overwrites(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sdk, "register_team", _fake_register())
    target = tmp_path / "teamx.kubeconfig"
    target.write_text("OLD-CONTENT", encoding="utf-8")
    result = runner.invoke(
        cli.app, ["register", "teamx", "--out", str(target), "--force"]
    )
    assert result.exit_code == 0
    assert target.read_text(encoding="utf-8") == "KUBECONFIG-DATA\n"


def test_cli_print_writes_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sdk, "register_team", _fake_register())
    target = tmp_path / "teamx.kubeconfig"
    result = runner.invoke(
        cli.app, ["register", "teamx", "--print", "--out", str(target)]
    )
    assert result.exit_code == 0
    assert "KUBECONFIG-DATA" in result.output
    assert not target.exists()


def test_cli_forwards_apiserver_and_storage(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli.sdk, "register_team",
        lambda team, **kw: seen.update(kw) or _fake_register()(team, **kw),
    )
    result = runner.invoke(
        cli.app,
        ["register", "teamx", "--print", "--apiserver", "https://5.6.7.8:6443",
         "--storage", "20Ti"],
    )
    assert result.exit_code == 0
    assert seen.get("apiserver") == "https://5.6.7.8:6443"
    assert seen.get("storage") == "20Ti"


def test_cli_wuji_error_friendly_exit(monkeypatch, tmp_path):
    def boom(team, **kw):
        raise sdk.WujiError("权限不足:请用管理员 kubeconfig")

    monkeypatch.setattr(cli.sdk, "register_team", boom)
    target = tmp_path / "teamx.kubeconfig"
    result = runner.invoke(cli.app, ["register", "teamx", "--out", str(target)])
    assert result.exit_code == 1
    assert "权限不足" in result.output
    assert not target.exists()
    # friendly exit — no leaked traceback / WujiError propagation
    assert not isinstance(result.exception, sdk.WujiError)


def test_cli_reports_reused_when_nothing_created(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.sdk, "register_team", _fake_register(created=[]))
    target = tmp_path / "teamx.kubeconfig"
    result = runner.invoke(cli.app, ["register", "teamx", "--out", str(target)])
    assert result.exit_code == 0
    assert "复用" in result.output
