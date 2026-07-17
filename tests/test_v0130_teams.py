"""v0.13.0 tests: ``volcano.sdk.list_teams`` + the ``teams`` CLI command.

Fully mocked (no real cluster / kubeconfig). Two layers:

* ``list_teams`` (SDK): lists namespaces carrying the ``wuji.io/team`` label
  (``list_namespace(label_selector=TEAM_LABEL)``), reads each ns's ``gpu-quota``
  ResourceQuota GPU hard limit from a cluster-wide RQ list (degrades to "-" when
  absent or unreadable), and folds in live GPU/pod counts from
  ``usage_by_namespace(None)``. Rows are sorted by namespace.
* ``teams`` (CLI, alias ``tm``): prints the table, an empty-state hint, and turns
  a ``WujiError`` from the SDK into a friendly non-zero exit.

Mocking: monkeypatch ``sdk.load_clients`` to hand back a fake CoreV1Api exposing
``list_namespace`` / ``list_resource_quota_for_all_namespaces``, and patch
``sdk.usage_by_namespace``. Only tests/ is touched.
"""

from types import SimpleNamespace

import pytest
from kubernetes.client.exceptions import ApiException

import volcano.cli as cli
import volcano.sdk as sdk
from volcano.kube import TEAM_LABEL, WujiError
from typer.testing import CliRunner

runner = CliRunner()


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _ns_item(name, team):
    """A namespace object as list_teams reads it (metadata.name + labels)."""
    labels = {} if team is None else {TEAM_LABEL: team}
    return SimpleNamespace(metadata=SimpleNamespace(name=name, labels=labels))


def _rq_item(namespace, name, gpu):
    """A ResourceQuota list item (metadata.name/namespace + spec.hard)."""
    hard = {} if gpu is None else {"requests.nvidia.com/gpu": gpu}
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, namespace=namespace),
        spec=SimpleNamespace(hard=hard),
    )


class _TeamsCore:
    def __init__(self, ns_items, *, rq_items=None, rq_exc=None, ns_exc=None):
        self.ns_items = ns_items
        self.rq_items = rq_items or []
        self.rq_exc = rq_exc
        self.ns_exc = ns_exc
        self.ns_selector = "<unset>"

    def list_namespace(self, label_selector=None):
        self.ns_selector = label_selector
        if self.ns_exc:
            raise self.ns_exc
        return SimpleNamespace(items=self.ns_items)

    def list_resource_quota_for_all_namespaces(self):
        if self.rq_exc:
            raise self.rq_exc
        return SimpleNamespace(items=self.rq_items)


def _wire_teams(monkeypatch, core, *, usage=None, usage_exc=None):
    """load_clients -> (core, None, None); usage_by_namespace -> usage (or raises).

    Records the argument list_teams passes to usage_by_namespace in ``seen``.
    """
    seen = {}
    monkeypatch.setattr(sdk, "load_clients", lambda: (core, None, None))

    def fake_usage(team=None):
        seen["arg"] = team
        if usage_exc is not None:
            raise usage_exc
        return usage or []

    monkeypatch.setattr(sdk, "usage_by_namespace", fake_usage)
    return seen


# =========================================================================== #
# list_teams — SDK
# =========================================================================== #
def test_list_teams_shape_selector_and_sort(monkeypatch):
    core = _TeamsCore(
        # returned out of namespace order to prove list_teams sorts them
        [_ns_item("bob", "blueteam"), _ns_item("alice", "redteam")],
        rq_items=[_rq_item("alice", "gpu-quota", "8")],
    )
    seen = _wire_teams(monkeypatch, core, usage=[
        {"namespace": "alice", "gpu": 3, "pods": 2},
        {"namespace": "bob", "gpu": 0, "pods": 1},
    ])
    out = sdk.list_teams()
    # filtered by the team label + usage aggregated over all namespaces
    assert core.ns_selector == TEAM_LABEL
    assert seen["arg"] is None
    assert out == [
        {"namespace": "alice", "team": "redteam", "gpu_quota": "8", "gpu_used": 3, "pods": 2},
        {"namespace": "bob", "team": "blueteam", "gpu_quota": "-", "gpu_used": 0, "pods": 1},
    ]


def test_list_teams_gpu_quota_only_from_gpu_quota_rq(monkeypatch):
    # A differently-named RQ in the same ns must NOT be read as the gpu quota;
    # only the one literally named "gpu-quota" counts.
    core = _TeamsCore(
        [_ns_item("alice", "redteam")],
        rq_items=[
            _rq_item("alice", "some-other-quota", "99"),
            _rq_item("alice", "gpu-quota", "16"),
        ],
    )
    _wire_teams(monkeypatch, core)
    out = sdk.list_teams()
    assert out[0]["gpu_quota"] == "16"


def test_list_teams_gpu_quota_rq_without_gpu_key_is_dash(monkeypatch):
    # gpu-quota exists but only limits pods -> no requests.nvidia.com/gpu -> "-".
    core = _TeamsCore(
        [_ns_item("alice", "redteam")],
        rq_items=[
            SimpleNamespace(
                metadata=SimpleNamespace(name="gpu-quota", namespace="alice"),
                spec=SimpleNamespace(hard={"pods": "10"}),
            )
        ],
    )
    _wire_teams(monkeypatch, core)
    assert sdk.list_teams()[0]["gpu_quota"] == "-"


def test_list_teams_rq_list_apiexception_all_dash(monkeypatch):
    # Not permitted to list quotas cluster-wide -> quota column degrades to "-",
    # but namespaces/usage still resolve.
    core = _TeamsCore(
        [_ns_item("alice", "redteam"), _ns_item("bob", "blueteam")],
        rq_exc=ApiException(status=403),
    )
    _wire_teams(monkeypatch, core, usage=[{"namespace": "alice", "gpu": 5, "pods": 1}])
    out = sdk.list_teams()
    assert [r["gpu_quota"] for r in out] == ["-", "-"]
    assert out[0]["gpu_used"] == 5 and out[0]["pods"] == 1


def test_list_teams_usage_maps_used_and_defaults_zero(monkeypatch):
    # ns present in usage -> its gpu/pods; ns absent from usage -> 0/0.
    core = _TeamsCore([_ns_item("alice", "redteam"), _ns_item("bob", "blueteam")])
    _wire_teams(monkeypatch, core, usage=[{"namespace": "alice", "gpu": 7, "pods": 4}])
    out = sdk.list_teams()
    by_ns = {r["namespace"]: r for r in out}
    assert by_ns["alice"]["gpu_used"] == 7 and by_ns["alice"]["pods"] == 4
    assert by_ns["bob"]["gpu_used"] == 0 and by_ns["bob"]["pods"] == 0


def test_list_teams_usage_exception_swallowed(monkeypatch):
    # usage is best-effort: a failing usage_by_namespace must not crash list_teams;
    # gpu_used/pods just fall back to 0.
    core = _TeamsCore([_ns_item("alice", "redteam")], rq_items=[_rq_item("alice", "gpu-quota", "8")])
    _wire_teams(monkeypatch, core, usage_exc=RuntimeError("boom"))
    out = sdk.list_teams()
    assert out == [
        {"namespace": "alice", "team": "redteam", "gpu_quota": "8", "gpu_used": 0, "pods": 0}
    ]


def test_list_teams_empty(monkeypatch):
    core = _TeamsCore([])
    _wire_teams(monkeypatch, core)
    assert sdk.list_teams() == []


def test_list_teams_list_namespace_apiexception_raises_wuji(monkeypatch):
    # The namespace list is the primary data source; if it fails, surface a
    # friendly WujiError (unlike quota/usage which degrade silently).
    core = _TeamsCore([], ns_exc=ApiException(status=403))
    _wire_teams(monkeypatch, core)
    with pytest.raises(WujiError):
        sdk.list_teams()


def test_list_teams_rq_for_other_namespace_ignored(monkeypatch):
    # gpu-quota RQ that belongs to a DIFFERENT ns must not leak into this team.
    core = _TeamsCore(
        [_ns_item("alice", "redteam")],
        rq_items=[_rq_item("carol", "gpu-quota", "32")],
    )
    _wire_teams(monkeypatch, core)
    assert sdk.list_teams()[0]["gpu_quota"] == "-"


# =========================================================================== #
# teams — CLI (alias: tm)
# =========================================================================== #
def _guard_no_kube(monkeypatch):
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )


_ROWS = [
    {"namespace": "alice", "team": "redteam", "gpu_quota": "8", "gpu_used": 3, "pods": 2},
    {"namespace": "bob", "team": None, "gpu_quota": "-", "gpu_used": 0, "pods": 1},
]


def test_teams_table_with_data(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(cli.sdk, "list_teams", lambda: _ROWS)
    result = runner.invoke(cli.app, ["teams"])
    assert result.exit_code == 0, result.output
    # headers
    for header in ("命名空间", "团队", "GPU配额", "GPU已用", "Pods"):
        assert header in result.output
    # data cells
    assert "alice" in result.output and "redteam" in result.output
    assert "bob" in result.output


def test_teams_none_team_rendered_as_dash(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(
        cli.sdk, "list_teams",
        lambda: [{"namespace": "bob", "team": None, "gpu_quota": "-", "gpu_used": 0, "pods": 1}],
    )
    result = runner.invoke(cli.app, ["teams"])
    assert result.exit_code == 0, result.output
    assert "bob" in result.output
    assert "-" in result.output  # team None -> "-"


def test_teams_empty_prints_hint(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(cli.sdk, "list_teams", lambda: [])
    result = runner.invoke(cli.app, ["teams"])
    assert result.exit_code == 0, result.output
    assert "还没有带 wuji.io/team 标签的团队" in result.output


def test_teams_tm_alias_equivalent(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(cli.sdk, "list_teams", lambda: _ROWS)
    result = runner.invoke(cli.app, ["tm"])
    assert result.exit_code == 0, result.output
    assert "命名空间" in result.output
    assert "alice" in result.output and "bob" in result.output


def test_teams_wuji_error_friendly_exit(monkeypatch):
    _guard_no_kube(monkeypatch)

    def boom():
        raise cli.WujiError("列团队失败:权限不足")

    monkeypatch.setattr(cli.sdk, "list_teams", boom)
    result = runner.invoke(cli.app, ["teams"])
    assert result.exit_code == 1
    assert "列团队失败" in result.output
    assert not isinstance(result.exception, cli.WujiError)  # no leaked traceback
