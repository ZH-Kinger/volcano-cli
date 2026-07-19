"""v0.13.0 tests: ``volcano.sdk.list_teams`` + the ``teams`` CLI command.

Fully mocked (no real cluster / kubeconfig). Two layers:

* ``list_teams`` (SDK): aggregates namespaces carrying the ``wuji.io/team``
  label BY TEAM (ns=person, queue=team, so several people's namespaces can
  share one team) — one row per team, not per namespace. Namespaces missing
  the label are never merged with each other. GPU quota/usage/pods are summed
  across a team's member namespaces; each member's home-node (a per-namespace
  Gatekeeper Assign, punchlist §8) is reported individually since a team
  itself has no single home-node.
* ``teams`` (CLI, alias ``tm``): prints the aggregated table, an empty-state
  hint, and turns a ``WujiError`` from the SDK into a friendly non-zero exit.

Mocking: monkeypatch ``sdk.load_clients`` to hand back a fake CoreV1Api exposing
``list_namespace`` / ``list_resource_quota_for_all_namespaces``, and patch
``sdk.usage_by_namespace`` / ``sdk.get_home_node``. Only tests/ is touched.
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


def _wire_teams(monkeypatch, core, *, usage=None, usage_exc=None, home_nodes=None, home_node_exc_for=None):
    """load_clients -> (core, None, None); usage_by_namespace -> usage (or raises);
    get_home_node(ns) -> home_nodes.get(ns) (or raises for namespaces named in
    ``home_node_exc_for``, to prove a single bad member doesn't crash the call).

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

    home_nodes = home_nodes or {}
    home_node_exc_for = home_node_exc_for or set()
    home_node_calls = []

    def fake_get_home_node(namespace):
        home_node_calls.append(namespace)
        if namespace in home_node_exc_for:
            raise WujiError("权限不足")
        return home_nodes.get(namespace)

    monkeypatch.setattr(sdk, "get_home_node", fake_get_home_node)
    seen["home_node_calls"] = home_node_calls
    return seen


# =========================================================================== #
# list_teams — SDK
# =========================================================================== #
def test_list_teams_aggregates_by_team_label(monkeypatch):
    core = _TeamsCore(
        # two namespaces sharing one team, plus a lone one on another team,
        # returned out of order to prove list_teams sorts both groups and members
        [_ns_item("bob", "redteam"), _ns_item("carol", "blueteam"), _ns_item("alice", "redteam")],
        rq_items=[_rq_item("alice", "gpu-quota", "8"), _rq_item("bob", "gpu-quota", "4")],
    )
    seen = _wire_teams(
        monkeypatch, core,
        usage=[
            {"namespace": "alice", "gpu": 3, "pods": 2},
            {"namespace": "bob", "gpu": 1, "pods": 1},
            {"namespace": "carol", "gpu": 0, "pods": 0},
        ],
    )
    out = sdk.list_teams()
    assert core.ns_selector == TEAM_LABEL
    assert seen["arg"] is None
    # one row per TEAM, not per namespace: redteam merges alice+bob
    assert out == [
        {
            "team": "blueteam", "namespaces": ["carol"],
            "gpu_quota": "-", "gpu_used": 0, "pods": 0,
            "home_nodes": {"carol": None},
        },
        {
            "team": "redteam", "namespaces": ["alice", "bob"],
            "gpu_quota": "12", "gpu_used": 4, "pods": 3,
            "home_nodes": {"alice": None, "bob": None},
        },
    ]


def test_list_teams_untagged_namespaces_not_merged(monkeypatch):
    # two namespaces both missing the team label must NOT collapse into one
    # row -- they don't actually share a team.
    core = _TeamsCore([_ns_item("dave", None), _ns_item("erin", None)])
    _wire_teams(monkeypatch, core)
    out = sdk.list_teams()
    assert [r["namespaces"] for r in out] == [["dave"], ["erin"]]
    assert all(r["team"] == "" for r in out)


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


def test_list_teams_gpu_quota_partial_member_coverage_sums_known(monkeypatch):
    # alice has a readable quota, bob's is unreadable ("-") -> team sum counts
    # only alice's, bob doesn't zero it out or blank the whole team.
    core = _TeamsCore(
        [_ns_item("alice", "redteam"), _ns_item("bob", "redteam")],
        rq_items=[_rq_item("alice", "gpu-quota", "8")],
    )
    _wire_teams(monkeypatch, core)
    out = sdk.list_teams()
    assert out[0]["gpu_quota"] == "8"


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
    by_team = {r["team"]: r for r in out}
    assert by_team["redteam"]["gpu_used"] == 5 and by_team["redteam"]["pods"] == 1


def test_list_teams_usage_summed_across_team_members(monkeypatch):
    core = _TeamsCore([_ns_item("alice", "redteam"), _ns_item("bob", "redteam")])
    _wire_teams(monkeypatch, core, usage=[
        {"namespace": "alice", "gpu": 7, "pods": 4},
        {"namespace": "bob", "gpu": 2, "pods": 1},
    ])
    out = sdk.list_teams()
    assert out[0]["gpu_used"] == 9 and out[0]["pods"] == 5


def test_list_teams_usage_absent_member_defaults_zero(monkeypatch):
    # bob present in the namespace list but absent from usage -> contributes 0.
    core = _TeamsCore([_ns_item("alice", "redteam"), _ns_item("bob", "redteam")])
    _wire_teams(monkeypatch, core, usage=[{"namespace": "alice", "gpu": 7, "pods": 4}])
    out = sdk.list_teams()
    assert out[0]["gpu_used"] == 7 and out[0]["pods"] == 4


def test_list_teams_usage_exception_swallowed(monkeypatch):
    # usage is best-effort: a failing usage_by_namespace must not crash list_teams;
    # gpu_used/pods just fall back to 0.
    core = _TeamsCore([_ns_item("alice", "redteam")], rq_items=[_rq_item("alice", "gpu-quota", "8")])
    _wire_teams(monkeypatch, core, usage_exc=RuntimeError("boom"))
    out = sdk.list_teams()
    assert out == [
        {
            "team": "redteam", "namespaces": ["alice"],
            "gpu_quota": "8", "gpu_used": 0, "pods": 0,
            "home_nodes": {"alice": None},
        }
    ]


def test_list_teams_home_nodes_per_member(monkeypatch):
    # a team's members can each be pinned to a different node, or none.
    core = _TeamsCore([_ns_item("alice", "redteam"), _ns_item("bob", "redteam")])
    seen = _wire_teams(
        monkeypatch, core,
        home_nodes={"alice": "wuji-0"},  # bob has none set
    )
    out = sdk.list_teams()
    assert out[0]["home_nodes"] == {"alice": "wuji-0", "bob": None}
    assert sorted(seen["home_node_calls"]) == ["alice", "bob"]


def test_list_teams_home_node_error_for_one_member_degrades_not_crashes(monkeypatch):
    # get_home_node raising for one member (e.g. no RBAC to read Assigns) must
    # not blow up the whole call -- that member just reports None.
    core = _TeamsCore([_ns_item("alice", "redteam"), _ns_item("bob", "redteam")])
    _wire_teams(
        monkeypatch, core,
        home_nodes={"alice": "wuji-0"},
        home_node_exc_for={"bob"},
    )
    out = sdk.list_teams()
    assert out[0]["home_nodes"] == {"alice": "wuji-0", "bob": None}


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


def test_list_teams_sorted_by_team_then_by_first_member(monkeypatch):
    core = _TeamsCore(
        [_ns_item("zed", None), _ns_item("bob", "zteam"), _ns_item("alice", "ateam")],
    )
    _wire_teams(monkeypatch, core)
    out = sdk.list_teams()
    # untagged ("") sorts before any named team; ateam before zteam
    assert [r["team"] for r in out] == ["", "ateam", "zteam"]


# =========================================================================== #
# teams — CLI (alias: tm)
# =========================================================================== #
def _guard_no_kube(monkeypatch):
    monkeypatch.setattr(
        cli, "load_clients",
        lambda: (_ for _ in ()).throw(AssertionError("load_clients must not be called")),
    )


_ROWS = [
    {
        "team": "redteam", "namespaces": ["alice", "carol"],
        "gpu_quota": "8", "gpu_used": 3, "pods": 2,
        "home_nodes": {"alice": "wuji-0", "carol": None},
    },
    {
        "team": "", "namespaces": ["bob"],
        "gpu_quota": "-", "gpu_used": 0, "pods": 1,
        "home_nodes": {"bob": None},
    },
]


def test_teams_table_with_data(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(cli.sdk, "list_teams", lambda: _ROWS)
    result = runner.invoke(cli.app, ["teams"])
    assert result.exit_code == 0, result.output
    # headers
    for header in ("团队", "成员 namespace", "GPU配额", "GPU已用", "Pods", "主机软亲和"):
        assert header in result.output
    # data cells: members joined, home-node rendered as ns→node
    assert "redteam" in result.output
    assert "alice" in result.output and "carol" in result.output
    assert "alice→wuji-0" in result.output


def test_teams_home_node_dash_when_none_set(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(
        cli.sdk, "list_teams",
        lambda: [{
            "team": "blueteam", "namespaces": ["dave"],
            "gpu_quota": "-", "gpu_used": 0, "pods": 0,
            "home_nodes": {"dave": None},
        }],
    )
    result = runner.invoke(cli.app, ["teams"])
    assert result.exit_code == 0, result.output
    assert "dave" in result.output
    assert "-" in result.output  # no home-node set -> "-"


def test_teams_none_team_rendered_as_dash(monkeypatch):
    _guard_no_kube(monkeypatch)
    monkeypatch.setattr(
        cli.sdk, "list_teams",
        lambda: [{
            "team": "", "namespaces": ["bob"],
            "gpu_quota": "-", "gpu_used": 0, "pods": 1,
            "home_nodes": {"bob": None},
        }],
    )
    result = runner.invoke(cli.app, ["teams"])
    assert result.exit_code == 0, result.output
    assert "bob" in result.output
    assert "-" in result.output  # team "" -> "-"


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
    assert "团队" in result.output
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


# =========================================================================== #
# _format_home_nodes — CLI formatting helper
# =========================================================================== #
def test_format_home_nodes_joins_only_members_with_a_node_set():
    assert cli._format_home_nodes(
        ["alice", "bob", "carol"],
        {"alice": "wuji-0", "bob": None, "carol": "wuji-2"},
    ) == "alice→wuji-0, carol→wuji-2"


def test_format_home_nodes_all_unset_is_dash():
    assert cli._format_home_nodes(["alice"], {"alice": None}) == "-"
