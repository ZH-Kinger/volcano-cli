"""Unit tests for wuji.job.build_volcano_job — pure manifest logic, no cluster.

Covers: command wrapping (bash -lc vs -c), SSH prologue injection, --data /
--shared-data path resolution (leading-slash stripping) and env injection,
ports (ssh_port dedup + naming), --node nodeSelector, mount_path, conditional
datasets PVC mount, minAvailable=nodes / replicas, and the GPU limit.
"""

import shlex

import pytest

from wuji.job import (
    DATASETS_MOUNT,
    WORKSPACE_MOUNT,
    _DIST_PROLOGUE,
    _SSH_PROLOGUE,
    _normalize_command,
    _resolve_data_path,
    build_volcano_job,
)


def _base(**overrides):
    kwargs = dict(
        name="job1",
        team="teamx",
        queue="shared",
        image="acr/img:tag",
        command="python train.py",
    )
    kwargs.update(overrides)
    return build_volcano_job(**kwargs)


def _container(manifest):
    return manifest["spec"]["tasks"][0]["template"]["spec"]["containers"][0]


def _pod_spec(manifest):
    return manifest["spec"]["tasks"][0]["template"]["spec"]


# --------------------------------------------------------------------------- #
# _normalize_command
# --------------------------------------------------------------------------- #
def test_normalize_command_str_is_stripped():
    assert _normalize_command("  python train.py  ") == "python train.py"


def test_normalize_command_list_is_quoted_and_joined():
    out = _normalize_command(["python", "train.py", "--msg", "hello world"])
    assert out == "python train.py --msg 'hello world'"


# --------------------------------------------------------------------------- #
# command wrapping: bash -> -lc, others -> -c
# --------------------------------------------------------------------------- #
def test_bash_uses_login_flag():
    c = _container(_base(shell="bash"))
    assert c["command"][:2] == ["bash", "-lc"]


def test_non_bash_shell_uses_plain_c():
    c = _container(_base(shell="sh"))
    assert c["command"][:2] == ["sh", "-c"]


def test_command_string_carried_through():
    c = _container(_base(command="echo hi"))
    assert c["command"][2] == "echo hi"


# --------------------------------------------------------------------------- #
# SSH prologue injection
# --------------------------------------------------------------------------- #
def test_ssh_true_injects_prologue():
    c = _container(_base(ssh=True, command="python train.py"))
    body = c["command"][2]
    assert body.startswith(_SSH_PROLOGUE)
    assert body.endswith("python train.py")


def test_ssh_password_alone_triggers_prologue():
    # ssh=False but password given -> enable_ssh should be True.
    c = _container(_base(ssh=False, ssh_password="secret"))
    assert c["command"][2].startswith(_SSH_PROLOGUE)


def test_no_ssh_no_prologue():
    c = _container(_base())
    assert not c["command"][2].startswith(_SSH_PROLOGUE)
    assert c["command"][2] == "python train.py"


def test_ssh_env_injected():
    c = _container(_base(ssh=True, ssh_password="pw", ssh_port=2200))
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["WUJI_SSH_PASSWORD"] == "pw"
    assert env["WUJI_SSH_PORT"] == "2200"  # stringified


def test_ssh_password_none_becomes_empty_string():
    c = _container(_base(ssh=True))
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["WUJI_SSH_PASSWORD"] == ""


def test_ssh_pubkey_alone_triggers_prologue():
    c = _container(_base(ssh=False, ssh_pubkey="ssh-ed25519 AAAA user@host"))
    assert c["command"][2].startswith(_SSH_PROLOGUE)


def test_ssh_pubkey_env_injected():
    c = _container(_base(ssh=True, ssh_pubkey="ssh-rsa AAAAB3 user@host"))
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["WUJI_SSH_PUBKEY"] == "ssh-rsa AAAAB3 user@host"


def test_ssh_pubkey_empty_when_not_given():
    c = _container(_base(ssh=True))
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["WUJI_SSH_PUBKEY"] == ""


def test_prologue_generates_host_keys():
    # ssh-keygen -A is needed or sshd fails to start on slim images
    assert "ssh-keygen -A" in _SSH_PROLOGUE


def test_prologue_handles_pubkey_and_password_branches():
    # gated on either password OR pubkey being present
    assert '[ -n "$WUJI_SSH_PASSWORD" ] || [ -n "$WUJI_SSH_PUBKEY" ]' in _SSH_PROLOGUE
    # pubkey is appended to authorized_keys
    assert "authorized_keys" in _SSH_PROLOGUE
    # password auth only enabled when a password is set
    assert "PW_AUTH" in _SSH_PROLOGUE


# --------------------------------------------------------------------------- #
# _resolve_data_path + --data / --shared-data
# --------------------------------------------------------------------------- #
def test_resolve_strips_leading_slash():
    assert _resolve_data_path("/workspace", "/etc/passwd") == "/workspace/etc/passwd"


def test_resolve_empty_returns_mount_root():
    assert _resolve_data_path("/workspace", "   ") == "/workspace"
    assert _resolve_data_path("/workspace", "///") == "/workspace"


def test_data_resolves_under_workspace_and_injects_env():
    m = _base(data="runs/exp1")
    c = _container(m)
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["WUJI_DATA"] == "/workspace/runs/exp1"
    # value is appended to the command as a quoted --data arg
    assert "--data /workspace/runs/exp1" in c["command"][2]


def test_data_leading_slash_does_not_escape_mount():
    m = _base(data="/absolute/path")
    env = {e["name"]: e["value"] for e in _container(m)["env"]}
    assert env["WUJI_DATA"] == "/workspace/absolute/path"


def test_data_uses_custom_mount_path():
    m = _base(data="d", workspace_path="/cache")
    env = {e["name"]: e["value"] for e in _container(m)["env"]}
    assert env["WUJI_DATA"] == "/cache/d"


def test_shared_data_resolves_under_datasets():
    m = _base(shared_data="imagenet")
    c = _container(m)
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["WUJI_SHARED_DATA"] == "/datasets/imagenet"
    assert "--shared-data /datasets/imagenet" in c["command"][2]


def test_data_arg_is_shell_quoted():
    m = _base(data="has space")
    body = _container(m)["command"][2]
    assert f"--data {shlex.quote('/workspace/has space')}" in body


# --------------------------------------------------------------------------- #
# datasets PVC mounted only when shared_data is used
# --------------------------------------------------------------------------- #
def test_datasets_pvc_absent_without_shared_data():
    m = _base()
    vol_names = {v["name"] for v in _pod_spec(m)["volumes"]}
    assert "datasets" not in vol_names
    mount_names = {vm["name"] for vm in _container(m)["volumeMounts"]}
    assert "datasets" not in mount_names


def test_datasets_pvc_present_with_shared_data():
    m = _base(shared_data="ds", shared_dataset_pvc="my-shared")
    vols = {v["name"]: v for v in _pod_spec(m)["volumes"]}
    assert "datasets" in vols
    assert vols["datasets"]["persistentVolumeClaim"]["claimName"] == "my-shared"
    assert vols["datasets"]["persistentVolumeClaim"]["readOnly"] is True
    dmount = [vm for vm in _container(m)["volumeMounts"] if vm["name"] == "datasets"][0]
    assert dmount["mountPath"] == DATASETS_MOUNT
    assert dmount["readOnly"] is True


# --------------------------------------------------------------------------- #
# ports (+ ssh_port dedup / naming)
# --------------------------------------------------------------------------- #
def test_no_ports_no_ports_key():
    assert "ports" not in _container(_base())


def test_extra_ports_listed_plain():
    c = _container(_base(ports=[6006, 8265]))
    assert c["ports"] == [{"containerPort": 6006}, {"containerPort": 8265}]


def test_ssh_port_appended_and_named_when_ssh():
    c = _container(_base(ssh=True, ports=[6006], ssh_port=22))
    assert {"containerPort": 6006} in c["ports"]
    assert {"containerPort": 22, "name": "ssh"} in c["ports"]


def test_ssh_port_not_duplicated_if_already_present():
    c = _container(_base(ssh=True, ports=[22], ssh_port=22))
    assert c["ports"] == [{"containerPort": 22, "name": "ssh"}]


def test_custom_ssh_port_named_ssh():
    c = _container(_base(ssh=True, ssh_port=2222, ports=[2222]))
    assert c["ports"] == [{"containerPort": 2222, "name": "ssh"}]


# --------------------------------------------------------------------------- #
# node -> nodeSelector, mount_path, GPU limit, gang scheduling
# --------------------------------------------------------------------------- #
def test_node_sets_hostname_selector():
    ps = _pod_spec(_base(node="wuji-3"))
    assert ps["nodeSelector"] == {"kubernetes.io/hostname": "wuji-3"}


def test_no_node_no_selector():
    assert "nodeSelector" not in _pod_spec(_base())


def test_mount_path_drives_workdir_and_ws_mount():
    m = _base(workspace_path="/cache")
    c = _container(m)
    assert c["workingDir"] == "/cache"
    ws = [vm for vm in c["volumeMounts"] if vm["name"] == "ws"][0]
    assert ws["mountPath"] == "/cache"


def test_gpu_limit_is_stringified():
    c = _container(_base(gpus=4))
    assert c["resources"]["limits"]["nvidia.com/gpu"] == "4"


def test_cpu_and_memory_limits():
    c = _container(_base(cpu="32", memory="128Gi"))
    lim = c["resources"]["limits"]
    assert lim["cpu"] == "32"
    assert lim["memory"] == "128Gi"


def test_min_available_and_replicas_track_nodes():
    m = _base(nodes=3)
    assert m["spec"]["minAvailable"] == 3
    assert m["spec"]["tasks"][0]["replicas"] == 3


def test_ws_pvc_named_after_team():
    m = _base(team="alpha")
    ws = [v for v in _pod_spec(m)["volumes"] if v["name"] == "ws"][0]
    assert ws["persistentVolumeClaim"]["claimName"] == "alpha-nas"


def test_pod_carries_job_label():
    m = _base(name="myjob")
    labels = m["spec"]["tasks"][0]["template"]["metadata"]["labels"]
    assert labels["wuji.io/job"] == "myjob"


def test_kind_label_added_when_given():
    for k in ("train", "dev"):
        labels = _base(kind=k)["metadata"]["labels"]
        assert labels["wuji.io/kind"] == k
        assert labels["app.kubernetes.io/managed-by"] == "wuji"


def test_kind_label_absent_when_none():
    labels = _base()["metadata"]["labels"]
    assert "wuji.io/kind" not in labels


def test_kind_label_absent_when_empty_string():
    labels = _base(kind="")["metadata"]["labels"]
    assert "wuji.io/kind" not in labels


def test_scheduler_and_queue_and_namespace():
    m = _base(queue="pilot", team="teamz")
    assert m["spec"]["schedulerName"] == "volcano"
    assert m["spec"]["queue"] == "pilot"
    assert m["metadata"]["namespace"] == "teamz"


def test_dshm_tmpfs_present():
    m = _base(dshm_size="32Gi")
    dshm = [v for v in _pod_spec(m)["volumes"] if v["name"] == "dshm"][0]
    assert dshm["emptyDir"]["medium"] == "Memory"
    assert dshm["emptyDir"]["sizeLimit"] == "32Gi"


def test_default_mounts_workspace_constant():
    assert WORKSPACE_MOUNT == "/workspace"
    c = _container(_base())
    assert c["workingDir"] == WORKSPACE_MOUNT


# --------------------------------------------------------------------------- #
# multi-node distributed rendezvous (nodes > 1)
# --------------------------------------------------------------------------- #
def test_multinode_adds_svc_env_plugins():
    m = _base(nodes=2)
    assert m["spec"]["plugins"] == {"svc": [], "env": []}


def test_multinode_command_has_rendezvous_vars():
    body = _container(_base(nodes=2))["command"][2]
    assert _DIST_PROLOGUE in body
    for token in ("VC_WORKER_HOSTS", "MASTER_ADDR", "MASTER_PORT", "NNODES", "NODE_RANK"):
        assert token in body


def test_multinode_higher_replica_count():
    # plugins/prologue keyed on nodes>1, not a specific value
    m = _base(nodes=4)
    assert m["spec"]["plugins"] == {"svc": [], "env": []}
    assert _DIST_PROLOGUE in _container(m)["command"][2]


def test_single_node_no_plugins():
    assert "plugins" not in _base(nodes=1)["spec"]


def test_single_node_no_rendezvous_fragment():
    body = _container(_base(nodes=1))["command"][2]
    assert _DIST_PROLOGUE not in body
    assert "VC_WORKER_HOSTS" not in body
    assert "MASTER_ADDR" not in body


def test_dist_prologue_before_ssh_before_user_command():
    # order top-to-bottom: DIST rendezvous -> SSH prologue -> user command
    body = _container(_base(nodes=2, ssh=True, command="python train.py"))["command"][2]
    i_dist = body.index(_DIST_PROLOGUE)
    i_ssh = body.index(_SSH_PROLOGUE)
    i_user = body.index("python train.py")
    assert i_dist < i_ssh < i_user


def test_dist_prologue_precedes_user_command_without_ssh():
    body = _container(_base(nodes=2, command="python train.py"))["command"][2]
    assert body.index(_DIST_PROLOGUE) < body.index("python train.py")
    assert not body.startswith(_SSH_PROLOGUE)  # no SSH when not requested


def test_dist_prologue_content():
    # ${VAR:-default} form + first-host extraction + rank from VC_TASK_INDEX
    assert "MASTER_PORT:-29500" in _DIST_PROLOGUE
    assert "NNODES:-${VC_WORKER_NUM:-1}" in _DIST_PROLOGUE
    assert "NODE_RANK:-${VC_TASK_INDEX:-0}" in _DIST_PROLOGUE
    assert "cut -d, -f1" in _DIST_PROLOGUE  # first host as MASTER_ADDR
