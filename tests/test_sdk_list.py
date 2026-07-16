"""Unit tests for volcano.sdk.list_jobs kind filtering + kind field.

The kubernetes CustomObjects client is mocked — no cluster.
"""

import pytest

import volcano.sdk as sdk


def _item(name, kind=None, gpus="8", replicas=1):
    labels = {"app.kubernetes.io/managed-by": "wuji"}
    if kind:
        labels["wuji.io/kind"] = kind
    return {
        "metadata": {"name": name, "labels": labels, "creationTimestamp": "2026-01-01T00:00:00Z"},
        "spec": {
            "queue": "shared",
            "tasks": [
                {
                    "replicas": replicas,
                    "template": {
                        "spec": {
                            "containers": [
                                {"resources": {"limits": {"nvidia.com/gpu": gpus}}}
                            ]
                        }
                    },
                }
            ],
        },
        "status": {"state": {"phase": "Running"}},
    }


@pytest.fixture
def fake_items(monkeypatch):
    box = {"items": []}

    class Custom:
        def list_namespaced_custom_object(self, **kw):
            return {"items": box["items"]}

    monkeypatch.setattr(sdk, "load_clients", lambda: (None, None, Custom()))
    return box


def test_kind_field_present_and_defaults_to_dash(fake_items):
    fake_items["items"] = [_item("a", kind="train"), _item("b")]  # b has no kind label
    out = sdk.list_jobs(team="teamx")
    kinds = {j["name"]: j["kind"] for j in out}
    assert kinds == {"a": "train", "b": "-"}


def test_kind_filter_keeps_only_matching(fake_items):
    fake_items["items"] = [
        _item("a", kind="train"), _item("b", kind="dev"), _item("c")
    ]
    out = sdk.list_jobs(team="teamx", kind="train")
    assert [j["name"] for j in out] == ["a"]


def test_kind_filter_dev(fake_items):
    fake_items["items"] = [_item("a", kind="train"), _item("b", kind="dev")]
    out = sdk.list_jobs(team="teamx", kind="dev")
    assert [j["name"] for j in out] == ["b"]


def test_kind_filter_excludes_unlabeled(fake_items):
    fake_items["items"] = [_item("c")]  # no kind label
    assert sdk.list_jobs(team="teamx", kind="dev") == []
    assert sdk.list_jobs(team="teamx", kind="train") == []


def test_no_filter_returns_all(fake_items):
    fake_items["items"] = [_item("a", kind="train"), _item("b", kind="dev"), _item("c")]
    out = sdk.list_jobs(team="teamx")
    assert {j["name"] for j in out} == {"a", "b", "c"}


def test_summary_fields_carried(fake_items):
    fake_items["items"] = [_item("a", kind="train", gpus="8", replicas=2)]
    (j,) = sdk.list_jobs(team="teamx")
    assert j["name"] == "a"
    assert j["phase"] == "Running"
    assert j["queue"] == "shared"
    assert j["gpus"] == 16  # replicas 2 * 8
    assert j["kind"] == "train"
    assert j["age"] == "2026-01-01T00:00:00Z"


def test_empty_list(fake_items):
    fake_items["items"] = []
    assert sdk.list_jobs(team="teamx") == []
    assert sdk.list_jobs(team="teamx", kind="dev") == []
