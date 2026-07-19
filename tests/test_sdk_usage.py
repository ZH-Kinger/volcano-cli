"""Unit tests for the gpu-snapshot-backed top_nodes()/usage_by_namespace().

These used to read pods cluster-wide (`list_pod_for_all_namespaces`); they now
read each GPU node's `gpu-usage-<node>` ConfigMap (written by gpu-snapshot.py,
see gpu-snapshot.yaml for the RBAC). Regular users only ever get `get` on those
5 known ConfigMap names (verified live: k8s RBAC `resourceNames` doesn't extend
to list/watch), so there's no list-by-label fallback — the code fetches the
node list (already permitted) and does one `get` per node name, skipping any
node whose ConfigMap can't be read. The kubernetes client is fully mocked.
"""

import json
from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

import volcano.sdk as sdk


def _node(name, gpu_cap=8, cpu="64", mem="256Gi"):
    return {
        "metadata": {"name": name},
        "status": {"allocatable": {"nvidia.com/gpu": str(gpu_cap), "cpu": cpu, "memory": mem}},
    }


def _non_gpu_node(name="cn-huhehaote.172.23.115.116"):
    return {"metadata": {"name": name}, "status": {"allocatable": {"cpu": "8", "memory": "32Gi"}}}


class FakeCore:
    """Records calls; stands in for CoreV1Api. Nodes come back as raw JSON
    bytes (matches the real `_preload_content=False` usage); ConfigMaps come
    back as typed-ish objects with a `.data` dict, or raise per name."""

    def __init__(self, nodes=None, cm_data=None, cm_exc=None):
        self.nodes = nodes or []
        self.cm_data = cm_data or {}   # {name: {"k8s_pods.json": "..."}}
        self.cm_exc = cm_exc or {}     # {name: ApiException}
        self.requested_cms = []

    def list_node(self, _preload_content=False):
        body = json.dumps({"items": self.nodes}).encode()
        return SimpleNamespace(data=body)

    def read_namespaced_config_map(self, *, name, namespace):
        self.requested_cms.append((name, namespace))
        if name in self.cm_exc:
            raise self.cm_exc[name]
        if name not in self.cm_data:
            raise ApiException(status=404, reason="NotFound")
        return SimpleNamespace(data=self.cm_data[name])


@pytest.fixture
def fake_clients(monkeypatch):
    core = FakeCore()
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))
    return core


def _pods_json(*pods):
    return json.dumps(list(pods))


# --------------------------------------------------------------------------- #
# _list_gpu_nodes
# --------------------------------------------------------------------------- #
def test_list_gpu_nodes_filters_non_gpu(fake_clients):
    fake_clients.nodes = [_node("wuji-0"), _non_gpu_node()]
    out = sdk._list_gpu_nodes(fake_clients)
    assert [n["metadata"]["name"] for n in out] == ["wuji-0"]


def test_list_gpu_nodes_api_error_wrapped(monkeypatch, fake_clients):
    def boom(_preload_content=False):
        raise ApiException(status=403, reason="Forbidden")

    fake_clients.list_node = boom
    with pytest.raises(sdk.WujiError):
        sdk._list_gpu_nodes(fake_clients)


# --------------------------------------------------------------------------- #
# _gpu_snapshot_entries
# --------------------------------------------------------------------------- #
def test_gpu_snapshot_entries_one_get_per_node_tagged(fake_clients):
    fake_clients.cm_data = {
        "gpu-usage-wuji-0": {"k8s_pods.json": _pods_json(
            {"ns": "alice", "name": "job1", "gpus": 2, "cpu_m": 16000.0, "mem_gi": 64.0}
        )},
        "gpu-usage-wuji-1": {"k8s_pods.json": _pods_json(
            {"ns": "bob", "name": "job2", "gpus": 1, "cpu_m": 4000.0, "mem_gi": 8.0}
        )},
    }
    out = sdk._gpu_snapshot_entries(fake_clients, ["wuji-0", "wuji-1"])
    assert fake_clients.requested_cms == [
        ("gpu-usage-wuji-0", "wuji-system"), ("gpu-usage-wuji-1", "wuji-system"),
    ]
    assert {p["ns"] for p in out} == {"alice", "bob"}
    alice = next(p for p in out if p["ns"] == "alice")
    assert alice["node"] == "wuji-0" and alice["gpus"] == 2


def test_gpu_snapshot_entries_skips_unreadable_node(fake_clients):
    # wuji-0 not yet in the RBAC resourceNames list (or 404, hasn't published yet)
    fake_clients.cm_exc = {"gpu-usage-wuji-0": ApiException(status=403, reason="Forbidden")}
    fake_clients.cm_data = {
        "gpu-usage-wuji-1": {"k8s_pods.json": _pods_json({"ns": "bob", "gpus": 1})},
    }
    out = sdk._gpu_snapshot_entries(fake_clients, ["wuji-0", "wuji-1"])
    assert len(out) == 1
    assert out[0]["ns"] == "bob"


def test_gpu_snapshot_entries_missing_configmap_skipped(fake_clients):
    out = sdk._gpu_snapshot_entries(fake_clients, ["wuji-0"])  # no cm_data seeded -> 404
    assert out == []


def test_gpu_snapshot_entries_malformed_json_skipped(fake_clients):
    fake_clients.cm_data = {"gpu-usage-wuji-0": {"k8s_pods.json": "not json"}}
    out = sdk._gpu_snapshot_entries(fake_clients, ["wuji-0"])
    assert out == []


def test_gpu_snapshot_entries_empty_data_field(fake_clients):
    fake_clients.cm_data = {"gpu-usage-wuji-0": {}}  # no k8s_pods.json key at all
    out = sdk._gpu_snapshot_entries(fake_clients, ["wuji-0"])
    assert out == []


# --------------------------------------------------------------------------- #
# top_nodes()
# --------------------------------------------------------------------------- #
def test_top_nodes_combines_capacity_and_snapshot_usage(fake_clients):
    fake_clients.nodes = [_node("wuji-0", gpu_cap=8, cpu="64", mem="256Gi"),
                          _node("wuji-1", gpu_cap=8, cpu="64", mem="256Gi")]
    fake_clients.cm_data = {
        "gpu-usage-wuji-0": {"k8s_pods.json": _pods_json(
            {"ns": "alice", "gpus": 3, "cpu_m": 24000.0, "mem_gi": 96.0}
        )},
        # wuji-1 has no usage published yet -> should show zeros, not error
    }
    out = sdk.top_nodes()
    assert [n["node"] for n in out] == ["wuji-0", "wuji-1"]  # sorted
    n0 = out[0]
    assert n0["gpu_used"] == 3 and n0["gpu_cap"] == 8
    assert n0["cpu_used_m"] == 24000.0 and n0["mem_used_gi"] == 96.0
    n1 = out[1]
    assert n1["gpu_used"] == 0 and n1["gpu_cap"] == 8
    assert n1["cpu_used_m"] == 0.0 and n1["mem_used_gi"] == 0.0


def test_top_nodes_sums_multiple_pods_on_same_node(fake_clients):
    fake_clients.nodes = [_node("wuji-0")]
    fake_clients.cm_data = {
        "gpu-usage-wuji-0": {"k8s_pods.json": _pods_json(
            {"ns": "alice", "gpus": 2, "cpu_m": 8000.0, "mem_gi": 16.0},
            {"ns": "bob", "gpus": 1, "cpu_m": 4000.0, "mem_gi": 8.0},
        )},
    }
    out = sdk.top_nodes()
    assert out[0]["gpu_used"] == 3
    assert out[0]["cpu_used_m"] == 12000.0
    assert out[0]["mem_used_gi"] == 24.0


def test_top_nodes_excludes_non_gpu_nodes(fake_clients):
    fake_clients.nodes = [_node("wuji-0"), _non_gpu_node()]
    out = sdk.top_nodes()
    assert [n["node"] for n in out] == ["wuji-0"]


def test_top_nodes_unreadable_node_shows_zero_not_error(fake_clients):
    fake_clients.nodes = [_node("wuji-0")]
    fake_clients.cm_exc = {"gpu-usage-wuji-0": ApiException(status=403, reason="Forbidden")}
    out = sdk.top_nodes()
    assert out == [{
        "node": "wuji-0", "gpu_used": 0, "gpu_cap": 8,
        "cpu_used_m": 0.0, "cpu_cap_m": 64000.0,
        "mem_used_gi": 0.0, "mem_cap_gi": 256.0,
    }]


# --------------------------------------------------------------------------- #
# usage_by_namespace()
# --------------------------------------------------------------------------- #
def test_usage_by_namespace_aggregates_across_nodes(fake_clients):
    fake_clients.nodes = [_node("wuji-0"), _node("wuji-1")]
    fake_clients.cm_data = {
        "gpu-usage-wuji-0": {"k8s_pods.json": _pods_json(
            {"ns": "alice", "gpus": 2, "cpu_m": 8000.0, "mem_gi": 16.0},
        )},
        "gpu-usage-wuji-1": {"k8s_pods.json": _pods_json(
            {"ns": "alice", "gpus": 1, "cpu_m": 4000.0, "mem_gi": 8.0},
            {"ns": "bob", "gpus": 4, "cpu_m": 32000.0, "mem_gi": 64.0},
        )},
    }
    out = sdk.usage_by_namespace()
    # sorted by -gpu: bob(4) before alice(3)
    assert out == [
        {"namespace": "bob", "gpu": 4, "cpu_m": 32000.0, "mem_gi": 64.0, "pods": 1},
        {"namespace": "alice", "gpu": 3, "cpu_m": 12000.0, "mem_gi": 24.0, "pods": 2},
    ]


def test_usage_by_namespace_team_filter_ignores_gpu_zero_default(fake_clients):
    fake_clients.nodes = [_node("wuji-0")]
    fake_clients.cm_data = {
        "gpu-usage-wuji-0": {"k8s_pods.json": _pods_json({"ns": "alice", "gpus": 2})},
    }
    out = sdk.usage_by_namespace(team="alice")
    assert out == [{"namespace": "alice", "gpu": 2, "cpu_m": 0.0, "mem_gi": 0.0, "pods": 1}]
    assert sdk.usage_by_namespace(team="nobody-here") == []


def test_usage_by_namespace_default_view_drops_zero_gpu_namespaces(fake_clients):
    # gpu-snapshot's collector itself never records gpu==0 pods, but usage_by_namespace's
    # own default-view filter should still hold even if a zero ever slipped through.
    fake_clients.nodes = [_node("wuji-0")]
    fake_clients.cm_data = {
        "gpu-usage-wuji-0": {"k8s_pods.json": _pods_json({"ns": "alice", "gpus": 0})},
    }
    assert sdk.usage_by_namespace() == []


def test_usage_by_namespace_no_snapshot_data_is_empty(fake_clients):
    fake_clients.nodes = [_node("wuji-0")]
    assert sdk.usage_by_namespace() == []
