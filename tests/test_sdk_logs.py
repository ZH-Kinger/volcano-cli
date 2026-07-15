"""Unit tests for wuji.sdk.logs pod/worker target resolution.

Rules: explicit ``pod`` wins; else ``worker=N`` -> ``<name>-worker-<N>``; else the
first pod (via label selector). The kubernetes core client is mocked — no cluster.
"""

from types import SimpleNamespace

import pytest

import wuji.sdk as sdk


class FakeCore:
    def __init__(self):
        self.log_calls = []
        self.pod_items = []

    def read_namespaced_pod_log(self, *, name, namespace, tail_lines=None):
        self.log_calls.append(name)
        return f"log of {name}"

    def list_namespaced_pod(self, *, namespace, label_selector):
        return SimpleNamespace(items=self.pod_items)


@pytest.fixture
def core(monkeypatch):
    c = FakeCore()
    monkeypatch.setattr(sdk, "load_clients", lambda: (c, None, None))
    return c


def test_explicit_pod_wins_over_worker(core):
    out = sdk.logs("j", team="teamx", pod="explicit-pod", worker=2)
    assert core.log_calls == ["explicit-pod"]
    assert out == "log of explicit-pod"


def test_worker_resolves_to_worker_pod(core):
    sdk.logs("j", team="teamx", worker=2)
    assert core.log_calls == ["j-worker-2"]


def test_worker_zero_is_honored(core):
    # worker=0 is falsy-ish but must still resolve (not fall through to first pod)
    sdk.logs("j", team="teamx", worker=0)
    assert core.log_calls == ["j-worker-0"]


def test_default_first_pod(core):
    core.pod_items = [
        SimpleNamespace(metadata=SimpleNamespace(name="j-worker-0")),
        SimpleNamespace(metadata=SimpleNamespace(name="j-worker-1")),
    ]
    sdk.logs("j", team="teamx")
    assert core.log_calls == ["j-worker-0"]


def test_no_pods_raises(core):
    core.pod_items = []
    with pytest.raises(sdk.WujiError):
        sdk.logs("j", team="teamx")


def test_tail_passed_through(core):
    captured = {}

    def read(*, name, namespace, tail_lines=None):
        captured["tail"] = tail_lines
        return "x"

    core.read_namespaced_pod_log = read
    sdk.logs("j", team="teamx", worker=1, tail=50)
    assert captured["tail"] == 50
