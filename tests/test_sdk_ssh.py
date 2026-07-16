"""Unit tests for the SSH NodePort surface in volcano.sdk.

_create_ssh_service body structure (NodePort, selector, ports, pinned nodePort),
ssh_endpoint resolution (svc nodePort + pod's node + node public-ip annotation,
and the None / partial cases), submit(expose_ssh=...) wiring, and kill's
best-effort Service cleanup. The kubernetes client is fully mocked — no cluster.
"""

from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

import volcano.sdk as sdk


class FakeCore:
    """Records calls and returns pre-seeded objects; stand-in for CoreV1Api."""

    def __init__(self):
        self.created_services = []
        self.deleted_services = []
        self.create_service_result = None
        self.create_service_results = None  # optional per-call queue
        self.create_service_exc = None
        self.service_by_name = {}
        self.read_service_exc = None
        self.pods = SimpleNamespace(items=[])
        self.list_pod_exc = None
        self.nodes = {}
        self.read_node_exc = None
        # per-worker (increment 2-A) plumbing
        self.services_list = []          # items returned by list_namespaced_service
        self.list_service_exc = None
        self.list_service_selector = None
        self.pod_by_name = {}            # read_namespaced_pod
        self.replaced_endpoints = []     # (name, body)
        self.created_endpoints = []      # body
        self.replace_ep_exc = None
        self.create_ep_exc = None

    # --- create service ---
    def create_namespaced_service(self, *, namespace, body):
        self.created_services.append((namespace, body))
        if self.create_service_exc:
            raise self.create_service_exc
        if self.create_service_results is not None:
            return self.create_service_results.pop(0)
        return self.create_service_result

    # --- list services (kill cleanup, by label) ---
    def list_namespaced_service(self, *, namespace, label_selector):
        self.list_service_selector = label_selector
        if self.list_service_exc:
            raise self.list_service_exc
        return SimpleNamespace(items=self.services_list)

    # --- read pod by name (worker endpoint reconcile) ---
    def read_namespaced_pod(self, *, name, namespace):
        if name not in self.pod_by_name:
            raise ApiException(status=404, reason="NotFound")
        return self.pod_by_name[name]

    # --- endpoints upsert ---
    def replace_namespaced_endpoints(self, *, name, namespace, body):
        if self.replace_ep_exc:
            raise self.replace_ep_exc
        self.replaced_endpoints.append((name, body))
        return None

    def create_namespaced_endpoints(self, *, namespace, body):
        if self.create_ep_exc:
            raise self.create_ep_exc
        self.created_endpoints.append(body)
        return None

    # --- read service ---
    def read_namespaced_service(self, *, name, namespace):
        if self.read_service_exc:
            raise self.read_service_exc
        if name not in self.service_by_name:
            raise ApiException(status=404, reason="NotFound")
        return self.service_by_name[name]

    # --- pods ---
    def list_namespaced_pod(self, *, namespace, label_selector):
        if self.list_pod_exc:
            raise self.list_pod_exc
        return self.pods

    # --- node ---
    def read_node(self, *, name):
        if self.read_node_exc:
            raise self.read_node_exc
        if name not in self.nodes:
            raise ApiException(status=404, reason="NotFound")
        return self.nodes[name]

    # --- delete service ---
    def delete_namespaced_service(self, *, name, namespace):
        self.deleted_services.append((name, namespace))
        return None


@pytest.fixture
def fake_clients(monkeypatch):
    core = FakeCore()
    custom = SimpleNamespace()
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, custom))
    return core, custom


def _svc_with_nodeport(node_port):
    port = SimpleNamespace(node_port=node_port)
    return SimpleNamespace(spec=SimpleNamespace(ports=[port]))


# --------------------------------------------------------------------------- #
# _create_ssh_service
# --------------------------------------------------------------------------- #
def test_create_ssh_service_body_structure(fake_clients):
    core, _ = fake_clients
    core.create_service_result = _svc_with_nodeport(31000)

    allocated = sdk._create_ssh_service("job1", ns="teamx", ssh_port=22)

    assert allocated == 31000
    ns, body = core.created_services[0]
    assert ns == "teamx"
    assert body["kind"] == "Service"
    assert body["metadata"]["name"] == "job1-ssh"
    assert body["metadata"]["namespace"] == "teamx"
    spec = body["spec"]
    assert spec["type"] == "NodePort"
    assert spec["selector"] == {"wuji.io/job": "job1"}
    assert len(spec["ports"]) == 1
    p = spec["ports"][0]
    assert p["port"] == 22
    assert p["targetPort"] == 22
    assert p["protocol"] == "TCP"
    # auto-assigned -> no pinned nodePort in the request body
    assert "nodePort" not in p


def test_create_ssh_service_pins_node_port(fake_clients):
    core, _ = fake_clients
    core.create_service_result = _svc_with_nodeport(30500)

    sdk._create_ssh_service("job1", ns="teamx", ssh_port=2222, node_port=30500)

    _, body = core.created_services[0]
    p = body["spec"]["ports"][0]
    assert p["nodePort"] == 30500
    assert p["port"] == 2222
    assert p["targetPort"] == 2222


def test_create_ssh_service_custom_port_targetport(fake_clients):
    core, _ = fake_clients
    core.create_service_result = _svc_with_nodeport(32000)
    sdk._create_ssh_service("j", ns="n", ssh_port=2200)
    _, body = core.created_services[0]
    p = body["spec"]["ports"][0]
    assert p["port"] == 2200 and p["targetPort"] == 2200


def test_create_ssh_service_managed_by_label(fake_clients):
    core, _ = fake_clients
    core.create_service_result = _svc_with_nodeport(31000)
    sdk._create_ssh_service("job1", ns="teamx")
    _, body = core.created_services[0]
    labels = body["metadata"]["labels"]
    assert labels["app.kubernetes.io/managed-by"] == "wuji"
    assert labels["wuji.io/job"] == "job1"


def test_create_ssh_service_api_error_wrapped(fake_clients):
    core, _ = fake_clients
    core.create_service_exc = ApiException(status=409, reason="Conflict")
    with pytest.raises(sdk.WujiError) as ei:
        sdk._create_ssh_service("job1", ns="teamx")
    assert "SSH" in str(ei.value)


def test_create_ssh_service_no_ports_returns_none(fake_clients):
    core, _ = fake_clients
    core.create_service_result = SimpleNamespace(spec=SimpleNamespace(ports=None))
    assert sdk._create_ssh_service("job1", ns="teamx") is None


def test_create_ssh_service_owner_reference_added(fake_clients):
    core, _ = fake_clients
    core.create_service_result = _svc_with_nodeport(31000)
    owner = {"metadata": {"uid": "u1", "name": "job1"}}
    sdk._create_ssh_service("job1", ns="teamx", owner=owner)
    _, body = core.created_services[0]
    refs = body["metadata"]["ownerReferences"]
    assert len(refs) == 1
    ref = refs[0]
    assert ref["apiVersion"] == "batch.volcano.sh/v1alpha1"
    assert ref["kind"] == "Job"
    assert ref["name"] == "job1"
    assert ref["uid"] == "u1"
    assert ref["controller"] is False
    assert ref["blockOwnerDeletion"] is False


def test_create_ssh_service_no_owner_no_reference(fake_clients):
    core, _ = fake_clients
    core.create_service_result = _svc_with_nodeport(31000)
    sdk._create_ssh_service("job1", ns="teamx")
    _, body = core.created_services[0]
    assert "ownerReferences" not in body["metadata"]


def test_create_ssh_service_owner_without_uid_no_reference(fake_clients):
    core, _ = fake_clients
    core.create_service_result = _svc_with_nodeport(31000)
    # created object exists but the API server did not return a uid
    sdk._create_ssh_service("job1", ns="teamx", owner={"metadata": {}})
    _, body = core.created_services[0]
    assert "ownerReferences" not in body["metadata"]


# --------------------------------------------------------------------------- #
# ssh_endpoint
# --------------------------------------------------------------------------- #
def test_ssh_endpoint_full_resolution(fake_clients):
    core, _ = fake_clients
    core.service_by_name["job1-ssh"] = _svc_with_nodeport(31234)
    core.pods = SimpleNamespace(
        items=[
            SimpleNamespace(
                spec=SimpleNamespace(node_name="wuji-2"),
                status=SimpleNamespace(phase="Running"),
            )
        ]
    )
    core.nodes["wuji-2"] = SimpleNamespace(
        metadata=SimpleNamespace(annotations={"wuji.io/public-ip": "1.2.3.4"})
    )

    ep = sdk.ssh_endpoint("job1", team="teamx")
    assert ep == {"node": "wuji-2", "public_ip": "1.2.3.4", "node_port": 31234}


def test_ssh_endpoint_no_service_returns_none(fake_clients):
    core, _ = fake_clients  # nothing seeded -> read raises 404
    assert sdk.ssh_endpoint("job1", team="teamx") is None


def test_ssh_endpoint_no_nodeport_returns_none(fake_clients):
    core, _ = fake_clients
    core.service_by_name["job1-ssh"] = SimpleNamespace(
        spec=SimpleNamespace(ports=[SimpleNamespace(node_port=None)])
    )
    assert sdk.ssh_endpoint("job1", team="teamx") is None


def test_ssh_endpoint_pod_unscheduled_gives_null_node_and_ip(fake_clients):
    core, _ = fake_clients
    core.service_by_name["job1-ssh"] = _svc_with_nodeport(31000)
    core.pods = SimpleNamespace(items=[])  # no pods yet
    ep = sdk.ssh_endpoint("job1", team="teamx")
    assert ep == {"node": None, "public_ip": None, "node_port": 31000}


def test_ssh_endpoint_node_missing_annotation(fake_clients):
    core, _ = fake_clients
    core.service_by_name["job1-ssh"] = _svc_with_nodeport(31000)
    core.pods = SimpleNamespace(
        items=[
            SimpleNamespace(
                spec=SimpleNamespace(node_name="wuji-0"),
                status=SimpleNamespace(phase="Running"),
            )
        ]
    )
    core.nodes["wuji-0"] = SimpleNamespace(
        metadata=SimpleNamespace(annotations={})  # no public-ip
    )
    ep = sdk.ssh_endpoint("job1", team="teamx")
    assert ep["node"] == "wuji-0"
    assert ep["public_ip"] is None
    assert ep["node_port"] == 31000


def test_ssh_endpoint_prefers_running_pod(fake_clients):
    core, _ = fake_clients
    core.service_by_name["job1-ssh"] = _svc_with_nodeport(31000)
    core.pods = SimpleNamespace(
        items=[
            SimpleNamespace(
                spec=SimpleNamespace(node_name="pending-node"),
                status=SimpleNamespace(phase="Pending"),
            ),
            SimpleNamespace(
                spec=SimpleNamespace(node_name="running-node"),
                status=SimpleNamespace(phase="Running"),
            ),
        ]
    )
    core.nodes["running-node"] = SimpleNamespace(
        metadata=SimpleNamespace(annotations={"wuji.io/public-ip": "9.9.9.9"})
    )
    ep = sdk.ssh_endpoint("job1", team="teamx")
    assert ep["node"] == "running-node"
    assert ep["public_ip"] == "9.9.9.9"


def test_ssh_endpoint_falls_back_to_scheduled_non_running(fake_clients):
    core, _ = fake_clients
    core.service_by_name["job1-ssh"] = _svc_with_nodeport(31000)
    core.pods = SimpleNamespace(
        items=[
            SimpleNamespace(
                spec=SimpleNamespace(node_name="some-node"),
                status=SimpleNamespace(phase="Pending"),
            )
        ]
    )
    core.nodes["some-node"] = SimpleNamespace(
        metadata=SimpleNamespace(annotations={"wuji.io/public-ip": "5.5.5.5"})
    )
    ep = sdk.ssh_endpoint("job1", team="teamx")
    assert ep["node"] == "some-node"
    assert ep["public_ip"] == "5.5.5.5"


def test_ssh_endpoint_pod_list_error_is_tolerated(fake_clients):
    core, _ = fake_clients
    core.service_by_name["job1-ssh"] = _svc_with_nodeport(31000)
    core.list_pod_exc = ApiException(status=403, reason="Forbidden")
    ep = sdk.ssh_endpoint("job1", team="teamx")
    assert ep == {"node": None, "public_ip": None, "node_port": 31000}


# --------------------------------------------------------------------------- #
# submit wiring (expose_ssh)
# --------------------------------------------------------------------------- #
def test_submit_expose_ssh_forces_ssh_and_injects_node_port(monkeypatch, fake_clients):
    core, custom = fake_clients
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return {"kind": "Job"}

    def fake_create(**kwargs):
        return {"metadata": {"name": kwargs["body"]["kind"]}}

    monkeypatch.setattr(sdk, "build_volcano_job", fake_build)
    custom.create_namespaced_custom_object = fake_create
    monkeypatch.setattr(
        sdk,
        "_create_ssh_service",
        lambda name, *, ns, ssh_port, node_port, owner=None: 31555,
    )

    out = sdk.submit(
        name="j1",
        team="teamx",
        image="img",
        command="python x.py",
        expose_ssh=True,
        ssh=False,
    )
    # expose_ssh implies ssh=True passed down to the manifest builder
    assert captured["ssh"] is True
    assert out["_ssh_node_port"] == 31555


def test_submit_without_expose_ssh_no_service(monkeypatch, fake_clients):
    core, custom = fake_clients
    monkeypatch.setattr(sdk, "build_volcano_job", lambda **kw: {"kind": "Job"})
    custom.create_namespaced_custom_object = lambda **kw: {"ok": True}
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1

    monkeypatch.setattr(sdk, "_create_ssh_service", boom)
    out = sdk.submit(name="j1", team="teamx", image="img", command="c")
    assert called["n"] == 0
    assert "_ssh_node_port" not in out


def _env_of(manifest):
    """Extract the worker container's env as a name->value dict."""
    c = manifest["spec"]["tasks"][0]["template"]["spec"]["containers"][0]
    return {e["name"]: e["value"] for e in c.get("env", [])}


def test_submit_autogenerates_password_when_none_given(fake_clients):
    # real build_volcano_job (not mocked) so we can inspect the manifest env
    core, custom = fake_clients
    custom.create_namespaced_custom_object = lambda **kw: dict(kw["body"])
    out = sdk.submit(name="j", team="teamx", image="img", command="c", ssh=True)
    gen = out.get("_ssh_generated_password")
    assert gen  # non-empty
    env = _env_of(out)
    assert env["WUJI_SSH_PASSWORD"] == gen
    assert env["WUJI_SSH_PASSWORD"] != ""


def test_submit_expose_ssh_autogenerates_password(monkeypatch, fake_clients):
    core, custom = fake_clients
    custom.create_namespaced_custom_object = lambda **kw: dict(kw["body"])
    monkeypatch.setattr(
        sdk, "_create_ssh_service",
        lambda name, *, ns, ssh_port, node_port, owner=None: 31000,
    )
    out = sdk.submit(
        name="j", team="teamx", image="img", command="c", expose_ssh=True
    )
    assert out.get("_ssh_generated_password")
    assert _env_of(out)["WUJI_SSH_PASSWORD"] == out["_ssh_generated_password"]


def test_submit_pubkey_does_not_autogenerate(fake_clients):
    core, custom = fake_clients
    custom.create_namespaced_custom_object = lambda **kw: dict(kw["body"])
    out = sdk.submit(
        name="j", team="teamx", image="img", command="c",
        ssh=True, ssh_pubkey="ssh-rsa AAAAB3Nz user@host",
    )
    assert "_ssh_generated_password" not in out
    env = _env_of(out)
    assert env["WUJI_SSH_PUBKEY"] == "ssh-rsa AAAAB3Nz user@host"
    assert env["WUJI_SSH_PASSWORD"] == ""  # pubkey-only: no password


def test_submit_explicit_password_no_autogenerate(fake_clients):
    core, custom = fake_clients
    custom.create_namespaced_custom_object = lambda **kw: dict(kw["body"])
    out = sdk.submit(
        name="j", team="teamx", image="img", command="c",
        ssh=True, ssh_password="hunter2",
    )
    assert "_ssh_generated_password" not in out
    assert _env_of(out)["WUJI_SSH_PASSWORD"] == "hunter2"


def test_submit_no_ssh_no_password_generated(fake_clients):
    core, custom = fake_clients
    custom.create_namespaced_custom_object = lambda **kw: dict(kw["body"])
    out = sdk.submit(name="j", team="teamx", image="img", command="c")
    assert "_ssh_generated_password" not in out
    assert _env_of(out) == {}  # no ssh -> no env


def test_submit_passes_node_port_through(monkeypatch, fake_clients):
    core, custom = fake_clients
    monkeypatch.setattr(sdk, "build_volcano_job", lambda **kw: {"kind": "Job"})
    custom.create_namespaced_custom_object = lambda **kw: {}
    seen = {}

    def fake_svc(name, *, ns, ssh_port, node_port, owner=None):
        seen["node_port"] = node_port
        seen["ssh_port"] = ssh_port
        return node_port

    monkeypatch.setattr(sdk, "_create_ssh_service", fake_svc)
    sdk.submit(
        name="j1",
        team="teamx",
        image="img",
        command="c",
        expose_ssh=True,
        ssh_node_port=30777,
        ssh_port=2200,
    )
    assert seen == {"node_port": 30777, "ssh_port": 2200}


# --------------------------------------------------------------------------- #
# kill: best-effort SSH Service cleanup
# --------------------------------------------------------------------------- #
def test_kill_deletes_ssh_services_by_label(fake_clients):
    core, custom = fake_clients
    custom.delete_namespaced_custom_object = lambda **kw: None
    # both the single-node `<name>-ssh` and per-worker `<name>-ssh-<i>` match the label
    core.services_list = [
        SimpleNamespace(metadata=SimpleNamespace(name="job1-ssh")),
        SimpleNamespace(metadata=SimpleNamespace(name="job1-ssh-0")),
        SimpleNamespace(metadata=SimpleNamespace(name="job1-ssh-1")),
    ]
    sdk.kill("job1", team="teamx")
    assert core.list_service_selector == "wuji.io/job=job1"
    deleted = {n for n, ns in core.deleted_services}
    assert deleted == {"job1-ssh", "job1-ssh-0", "job1-ssh-1"}


def test_kill_no_services_no_error(fake_clients):
    core, custom = fake_clients
    custom.delete_namespaced_custom_object = lambda **kw: None
    core.services_list = []
    sdk.kill("job1", team="teamx")  # must not raise
    assert core.deleted_services == []


def test_kill_tolerates_per_service_delete_error(fake_clients):
    core, custom = fake_clients
    custom.delete_namespaced_custom_object = lambda **kw: None
    core.services_list = [SimpleNamespace(metadata=SimpleNamespace(name="job1-ssh"))]

    def raise_404(*, name, namespace):
        raise ApiException(status=404, reason="NotFound")

    core.delete_namespaced_service = raise_404
    sdk.kill("job1", team="teamx")  # should not raise


def test_kill_tolerates_list_error(fake_clients):
    core, custom = fake_clients
    custom.delete_namespaced_custom_object = lambda **kw: None
    core.list_service_exc = ApiException(status=403, reason="Forbidden")
    sdk.kill("job1", team="teamx")  # should not raise


# --------------------------------------------------------------------------- #
# increment 2-A: per-worker public SSH (multi-node --expose-ssh)
# --------------------------------------------------------------------------- #
def _worker_svc(node_port, target_port=22):
    port = SimpleNamespace(node_port=node_port, target_port=target_port)
    return SimpleNamespace(spec=SimpleNamespace(ports=[port]))


def _pod(ip=None, uid="p1", node=None):
    return SimpleNamespace(
        status=SimpleNamespace(pod_ip=ip),
        metadata=SimpleNamespace(uid=uid),
        spec=SimpleNamespace(node_name=node),
    )


# ---- submit wiring: single vs per-worker ---------------------------------- #
def test_submit_expose_multinode_creates_worker_services(monkeypatch, fake_clients):
    core, custom = fake_clients
    monkeypatch.setattr(sdk, "build_volcano_job", lambda **kw: {"kind": "Job"})
    custom.create_namespaced_custom_object = lambda **kw: {"metadata": {"uid": "u1"}}
    seen = {}

    def fake_workers(name, *, ns, nodes, ssh_port, owner):
        seen.update(name=name, nodes=nodes, ssh_port=ssh_port, owner=owner)
        return {0: 30001, 1: 30002}

    monkeypatch.setattr(sdk, "_create_worker_ssh_services", fake_workers)
    monkeypatch.setattr(
        sdk, "_create_ssh_service",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("single path must not run")),
    )
    out = sdk.submit(
        name="j1", team="teamx", image="img", command="c", expose_ssh=True, nodes=2
    )
    assert out["_ssh_worker_node_ports"] == {0: 30001, 1: 30002}
    assert "_ssh_node_port" not in out
    assert seen["nodes"] == 2
    # owner is the created Job object (carries the uid used for ownerReferences)
    assert seen["owner"]["metadata"]["uid"] == "u1"


def test_submit_expose_singlenode_uses_single_service(monkeypatch, fake_clients):
    core, custom = fake_clients
    monkeypatch.setattr(sdk, "build_volcano_job", lambda **kw: {"kind": "Job"})
    custom.create_namespaced_custom_object = lambda **kw: {}
    monkeypatch.setattr(
        sdk, "_create_ssh_service",
        lambda name, *, ns, ssh_port, node_port, owner=None: 31000,
    )
    monkeypatch.setattr(
        sdk, "_create_worker_ssh_services",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("multi path must not run")),
    )
    out = sdk.submit(
        name="j1", team="teamx", image="img", command="c", expose_ssh=True, nodes=1
    )
    assert out["_ssh_node_port"] == 31000
    assert "_ssh_worker_node_ports" not in out


# ---- _create_worker_ssh_services body ------------------------------------- #
def test_create_worker_ssh_services_bodies(fake_clients):
    core, _ = fake_clients
    core.create_service_results = [_svc_with_nodeport(30001), _svc_with_nodeport(30002)]
    out = sdk._create_worker_ssh_services(
        "j1", ns="teamx", nodes=2, ssh_port=22, owner={"metadata": {"uid": "u1"}}
    )
    assert out == {0: 30001, 1: 30002}
    assert len(core.created_services) == 2
    for i, (ns, body) in enumerate(core.created_services):
        assert ns == "teamx"
        assert body["metadata"]["name"] == f"j1-ssh-{i}"
        assert body["spec"]["type"] == "NodePort"
        assert "selector" not in body["spec"]  # selector-less, manual Endpoints
        labels = body["metadata"]["labels"]
        assert labels["app.kubernetes.io/managed-by"] == "wuji"
        assert labels["wuji.io/job"] == "j1"
        assert labels["wuji.io/worker"] == str(i)
        p = body["spec"]["ports"][0]
        assert p["port"] == 22 and p["targetPort"] == 22
        ref = body["metadata"]["ownerReferences"][0]
        assert ref["kind"] == "Job" and ref["uid"] == "u1" and ref["name"] == "j1"


def test_create_worker_ssh_services_no_owner_no_ref(fake_clients):
    core, _ = fake_clients
    core.create_service_results = [_svc_with_nodeport(30001)]
    sdk._create_worker_ssh_services("j1", ns="teamx", nodes=1, ssh_port=22, owner=None)
    _, body = core.created_services[0]
    assert "ownerReferences" not in body["metadata"]


def test_create_worker_ssh_services_custom_port(fake_clients):
    core, _ = fake_clients
    core.create_service_results = [_svc_with_nodeport(30001)]
    sdk._create_worker_ssh_services("j1", ns="teamx", nodes=1, ssh_port=2200, owner=None)
    _, body = core.created_services[0]
    p = body["spec"]["ports"][0]
    assert p["port"] == 2200 and p["targetPort"] == 2200


def test_create_worker_ssh_services_api_error_wrapped(fake_clients):
    core, _ = fake_clients
    core.create_service_exc = ApiException(status=409, reason="Conflict")
    with pytest.raises(sdk.WujiError):
        sdk._create_worker_ssh_services("j1", ns="teamx", nodes=2, ssh_port=22, owner=None)


# ---- ensure_worker_endpoint ----------------------------------------------- #
def test_ensure_worker_endpoint_replaces_to_pod_ip(fake_clients):
    core, _ = fake_clients
    core.pod_by_name["j1-worker-0"] = _pod(ip="10.0.0.5", uid="pod-uid")
    ip = sdk.ensure_worker_endpoint("j1", 0, team="teamx", ssh_port=22)
    assert ip == "10.0.0.5"
    name, body = core.replaced_endpoints[0]
    assert name == "j1-ssh-0"
    assert body["kind"] == "Endpoints"
    subset = body["subsets"][0]
    assert subset["addresses"][0]["ip"] == "10.0.0.5"
    assert subset["ports"][0]["port"] == 22
    ref = body["metadata"]["ownerReferences"][0]
    assert ref["kind"] == "Pod" and ref["uid"] == "pod-uid" and ref["name"] == "j1-worker-0"
    assert core.created_endpoints == []  # replace succeeded, no create fallback


def test_ensure_worker_endpoint_no_pod_ip_returns_none(fake_clients):
    core, _ = fake_clients
    core.pod_by_name["j1-worker-0"] = _pod(ip=None)
    assert sdk.ensure_worker_endpoint("j1", 0, team="teamx") is None
    assert core.replaced_endpoints == [] and core.created_endpoints == []


def test_ensure_worker_endpoint_pod_missing_returns_none(fake_clients):
    core, _ = fake_clients  # nothing seeded -> read 404
    assert sdk.ensure_worker_endpoint("j1", 0, team="teamx") is None


def test_ensure_worker_endpoint_replace_then_create_fallback(fake_clients):
    core, _ = fake_clients
    core.pod_by_name["j1-worker-1"] = _pod(ip="10.0.0.9", uid="u9")
    core.replace_ep_exc = ApiException(status=404, reason="NotFound")
    ip = sdk.ensure_worker_endpoint("j1", 1, team="teamx", ssh_port=2200)
    assert ip == "10.0.0.9"
    assert len(core.created_endpoints) == 1
    body = core.created_endpoints[0]
    assert body["metadata"]["name"] == "j1-ssh-1"
    assert body["subsets"][0]["ports"][0]["port"] == 2200


def test_ensure_worker_endpoint_both_fail_returns_none(fake_clients):
    core, _ = fake_clients
    core.pod_by_name["j1-worker-0"] = _pod(ip="10.0.0.9")
    core.replace_ep_exc = ApiException(status=403, reason="Forbidden")
    core.create_ep_exc = ApiException(status=403, reason="Forbidden")
    assert sdk.ensure_worker_endpoint("j1", 0, team="teamx") is None


def test_ensure_worker_endpoint_no_uid_no_owner_ref(fake_clients):
    core, _ = fake_clients
    core.pod_by_name["j1-worker-0"] = _pod(ip="10.0.0.5", uid=None)
    sdk.ensure_worker_endpoint("j1", 0, team="teamx")
    _, body = core.replaced_endpoints[0]
    assert "ownerReferences" not in body["metadata"]


# ---- worker_ssh_endpoint --------------------------------------------------- #
def test_worker_ssh_endpoint_full_resolution(fake_clients):
    core, _ = fake_clients
    core.service_by_name["j1-ssh-0"] = _worker_svc(30005, target_port=22)
    core.pod_by_name["j1-worker-0"] = _pod(ip="10.0.0.5", uid="pu", node="wuji-2")
    core.nodes["wuji-2"] = SimpleNamespace(
        metadata=SimpleNamespace(annotations={"wuji.io/public-ip": "1.2.3.4"})
    )
    ep = sdk.worker_ssh_endpoint("j1", 0, team="teamx")
    assert ep == {
        "worker": 0, "pod": "j1-worker-0", "node": "wuji-2",
        "public_ip": "1.2.3.4", "node_port": 30005,
    }


def test_worker_ssh_endpoint_no_service_returns_none(fake_clients):
    core, _ = fake_clients
    assert sdk.worker_ssh_endpoint("j1", 0, team="teamx") is None


def test_worker_ssh_endpoint_no_ports_returns_none(fake_clients):
    core, _ = fake_clients
    core.service_by_name["j1-ssh-0"] = SimpleNamespace(spec=SimpleNamespace(ports=[]))
    assert sdk.worker_ssh_endpoint("j1", 0, team="teamx") is None


def test_worker_ssh_endpoint_pod_missing_partial(fake_clients):
    core, _ = fake_clients
    core.service_by_name["j1-ssh-0"] = _worker_svc(30005)
    ep = sdk.worker_ssh_endpoint("j1", 0, team="teamx")
    assert ep["node"] is None and ep["public_ip"] is None
    assert ep["node_port"] == 30005
    assert ep["pod"] == "j1-worker-0" and ep["worker"] == 0


def test_worker_ssh_endpoint_node_missing_annotation(fake_clients):
    core, _ = fake_clients
    core.service_by_name["j1-ssh-0"] = _worker_svc(30005)
    core.pod_by_name["j1-worker-0"] = _pod(ip="10.0.0.5", node="wuji-0")
    core.nodes["wuji-0"] = SimpleNamespace(metadata=SimpleNamespace(annotations={}))
    ep = sdk.worker_ssh_endpoint("j1", 0, team="teamx")
    assert ep["node"] == "wuji-0" and ep["public_ip"] is None
    assert ep["node_port"] == 30005
