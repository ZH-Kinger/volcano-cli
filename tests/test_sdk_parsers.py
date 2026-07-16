"""Unit tests for volcano.sdk resource-quantity parsers + _job_gpus.

Covers _gpu, _cpu_to_m (m-suffix / integer x1000 / bad input -> 0),
_mem_to_gi (Ki/Mi/Gi/Ti + plain bytes + bad input), and _job_gpus
(replicas x per-pod gpus, summed across tasks).
"""

import pytest

from volcano.sdk import _cpu_to_m, _gpu, _job_gpus, _mem_to_gi


# --------------------------------------------------------------------------- #
# _gpu
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("8", 8),
        (4, 4),
        (None, 0),
        ("", 0),
        ("abc", 0),
        (0, 0),
    ],
)
def test_gpu(value, expected):
    assert _gpu(value) == expected


# --------------------------------------------------------------------------- #
# _cpu_to_m
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("16", 16000.0),
        (16, 16000.0),
        ("1232m", 1232.0),
        ("500m", 500.0),
        ("1.5", 1500.0),
        (None, 0.0),
        ("garbage", 0.0),
        ("m", 0.0),  # empty numeric before 'm'
    ],
)
def test_cpu_to_m(value, expected):
    assert _cpu_to_m(value) == expected


# --------------------------------------------------------------------------- #
# _mem_to_gi
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("1Gi", 1.0),
        ("1024Mi", 1.0),
        ("1048576Ki", 1.0),
        ("2Ti", 2048.0),
        (str(2 ** 30), 1.0),  # plain bytes
        (2 ** 30, 1.0),
        (None, 0.0),
        ("badGi", 0.0),
        ("notanumber", 0.0),
    ],
)
def test_mem_to_gi(value, expected):
    assert _mem_to_gi(value) == pytest.approx(expected)


def test_mem_decimal_units():
    # 1G (decimal) = 1e9 bytes -> 1e9 / 2^30 GiB
    assert _mem_to_gi("1G") == pytest.approx(1e9 / (2 ** 30))
    assert _mem_to_gi("1M") == pytest.approx(1e6 / (2 ** 30))


# --------------------------------------------------------------------------- #
# _job_gpus
# --------------------------------------------------------------------------- #
def _task(replicas, gpus):
    return {
        "replicas": replicas,
        "template": {
            "spec": {
                "containers": [
                    {"resources": {"limits": {"nvidia.com/gpu": gpus}}}
                ]
            }
        },
    }


def test_job_gpus_single_task():
    assert _job_gpus([_task(2, 8)]) == 16


def test_job_gpus_sums_tasks():
    assert _job_gpus([_task(2, 8), _task(1, 4)]) == 20


def test_job_gpus_replicas_defaults_to_one():
    task = {
        "template": {
            "spec": {"containers": [{"resources": {"limits": {"nvidia.com/gpu": "8"}}}]}
        }
    }
    assert _job_gpus([task]) == 8


def test_job_gpus_missing_limits_is_zero():
    task = {"replicas": 3, "template": {"spec": {"containers": [{}]}}}
    assert _job_gpus([task]) == 0


def test_job_gpus_no_containers():
    assert _job_gpus([{"replicas": 2, "template": {"spec": {"containers": []}}}]) == 0


def test_job_gpus_bad_gpu_value_ignored():
    assert _job_gpus([_task(2, "notanint")]) == 0


def test_job_gpus_multiple_containers_per_pod():
    task = {
        "replicas": 2,
        "template": {
            "spec": {
                "containers": [
                    {"resources": {"limits": {"nvidia.com/gpu": "8"}}},
                    {"resources": {"limits": {"nvidia.com/gpu": "1"}}},
                ]
            }
        },
    }
    assert _job_gpus([task]) == 18  # 2 * (8 + 1)


def test_job_gpus_empty():
    assert _job_gpus([]) == 0
