"""Build the Volcano Job manifest (as a plain dict) for a training run.

The single public entry point is :func:`build_volcano_job`. It encodes the
cluster conventions so users never have to hand-write YAML:

* ``schedulerName: volcano`` + a ``queue`` + gang scheduling via ``minAvailable``;
* the three standard mounts — team NAS at ``/workspace``, a shared read-only
  dataset PVC at ``/datasets``, and a 64Gi ``tmpfs`` at ``/dev/shm``
  (required by PyTorch multi-GPU / NCCL shared memory);
* no ``imagePullSecret`` — the namespace ``default`` ServiceAccount is already
  bound to ``acr-cred``.
"""

from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional, Union

WORKSPACE_MOUNT = "/workspace"
DATASETS_MOUNT = "/datasets"
DSHM_MOUNT = "/dev/shm"

DEFAULT_SHARED_DATASET_PVC = "shared-datasets"
DEFAULT_DSHM_SIZE = "64Gi"

Command = Union[str, List[str]]

# 提交时开 --ssh 会把这段前置到命令里:设 root 密码/公钥 + 起 sshd,再跑训练命令。
# 需要镜像自带 openssh-server;缺失时尽力用 apt 安装,失败也不影响训练(继续跑)。
# 有密码 → 开口令登录;只给公钥 → 关口令登录(仅密钥,更安全)。
# ssh-keygen -A 生成主机密钥,否则某些精简镜像上 sshd 会起不来。
_SSH_PROLOGUE = (
    'if [ -n "$WUJI_SSH_PASSWORD" ] || [ -n "$WUJI_SSH_PUBKEY" ]; then\n'
    '  command -v sshd >/dev/null 2>&1 || '
    '{ apt-get update >/dev/null 2>&1 && apt-get install -y openssh-server >/dev/null 2>&1; } || true\n'
    '  mkdir -p /run/sshd /var/run/sshd ~/.ssh 2>/dev/null || true\n'
    '  chmod 700 ~/.ssh 2>/dev/null || true\n'
    '  ssh-keygen -A 2>/dev/null || true\n'
    '  if [ -n "$WUJI_SSH_PASSWORD" ]; then '
    'echo "root:$WUJI_SSH_PASSWORD" | chpasswd 2>/dev/null || true; fi\n'
    '  if [ -n "$WUJI_SSH_PUBKEY" ]; then '
    'echo "$WUJI_SSH_PUBKEY" >> ~/.ssh/authorized_keys && '
    'chmod 600 ~/.ssh/authorized_keys 2>/dev/null || true; fi\n'
    '  PW_AUTH=no; [ -n "$WUJI_SSH_PASSWORD" ] && PW_AUTH=yes\n'
    '  /usr/sbin/sshd -p "${WUJI_SSH_PORT:-22}" '
    '-o PermitRootLogin=yes -o "PasswordAuthentication=$PW_AUTH" 2>/dev/null '
    '|| service ssh start 2>/dev/null || true\n'
    'fi'
)


# 多机(nodes>1)时前置到命令里:从 Volcano svc/env 插件注入的环境变量里,
# 算出分布式 rendezvous 需要的 MASTER_ADDR / NNODES / NODE_RANK,让 torchrun 开箱即用。
# 用 ${VAR:-default} 只在用户未自设时才写,不覆盖用户显式设的值。
# 依赖 Job spec.plugins 里开了 svc(给 VC_WORKER_HOSTS/VC_WORKER_NUM)+ env(给 VC_TASK_INDEX)。
_DIST_PROLOGUE = (
    'if [ -n "$VC_WORKER_HOSTS" ]; then\n'
    '  export MASTER_ADDR="${MASTER_ADDR:-$(echo "$VC_WORKER_HOSTS" | cut -d, -f1)}"\n'
    '  export MASTER_PORT="${MASTER_PORT:-29500}"\n'
    '  export NNODES="${NNODES:-${VC_WORKER_NUM:-1}}"\n'
    '  export NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-0}}"\n'
    '  echo "[volcano] dist: MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT '
    'NNODES=$NNODES NODE_RANK=$NODE_RANK"\n'
    'fi'
)


def _normalize_command(command: Command) -> str:
    """Return the user command as one shell string.

    Args:
        command: either a shell string, or an argv list (joined & quoted).

    Returns:
        A single command string suitable for ``bash -lc``.
    """
    if isinstance(command, str):
        return command.strip()
    return " ".join(shlex.quote(part) for part in command)


def _resolve_data_path(mount: str, rel: str) -> str:
    """Join a mount root with a user-supplied relative path, safely.

    Leading slashes on ``rel`` are stripped so ``--data /etc`` cannot escape
    the mount. The result is a clean absolute path inside the container.

    Args:
        mount: the mount root, e.g. ``/workspace``.
        rel: the user-supplied relative sub-path.

    Returns:
        The absolute in-container path.
    """
    rel = rel.strip().lstrip("/")
    if not rel:
        return mount
    return f"{mount}/{rel}"


def build_volcano_job(
    *,
    name: str,
    team: str,
    queue: str,
    image: str,
    gpus: int = 8,
    nodes: int = 1,
    command: Command,
    data: Optional[str] = None,
    shared_data: Optional[str] = None,
    cpu: str = "16",
    memory: str = "64Gi",
    shell: str = "bash",
    workspace_path: str = WORKSPACE_MOUNT,
    shared_dataset_pvc: str = DEFAULT_SHARED_DATASET_PVC,
    dshm_size: str = DEFAULT_DSHM_SIZE,
    ssh: bool = False,
    ssh_password: Optional[str] = None,
    ssh_pubkey: Optional[str] = None,
    ssh_port: int = 22,
    ports: Optional[List[int]] = None,
    node: Optional[str] = None,
    kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Volcano ``batch.volcano.sh/v1alpha1`` Job manifest dict.

    Args:
        name: job name (must be a valid k8s name).
        team: team = target namespace; the NAS PVC is ``<team>-nas``.
        queue: Volcano queue to submit into (e.g. ``shared`` / ``pilot``).
        image: full ACR image reference.
        gpus: GPUs per worker (``nvidia.com/gpu`` limit). Defaults to 8.
        nodes: number of worker replicas; also ``minAvailable`` (gang). Default 1.
        command: the training command (string or argv list).
        data: optional path under ``/workspace`` (relative to the team NAS).
        shared_data: optional path under ``/datasets`` (shared read-only PVC).
        cpu: CPU limit per worker. Defaults to ``"16"``.
        memory: memory limit per worker. Defaults to ``"64Gi"``.
        shared_dataset_pvc: name of the shared read-only dataset PVC.
        dshm_size: size limit of the ``/dev/shm`` tmpfs. Defaults to ``64Gi``.

    Returns:
        A dict ready to hand to ``CustomObjectsApi.create_namespaced_custom_object``.
    """
    cmd = _normalize_command(command)
    env: List[Dict[str, str]] = []
    extra_args: List[str] = []

    if data:
        abs_data = _resolve_data_path(workspace_path, data)
        extra_args += ["--data", shlex.quote(abs_data)]
        env.append({"name": "WUJI_DATA", "value": abs_data})

    if shared_data:
        abs_shared = _resolve_data_path(DATASETS_MOUNT, shared_data)
        extra_args += ["--shared-data", shlex.quote(abs_shared)]
        env.append({"name": "WUJI_SHARED_DATA", "value": abs_shared})

    if extra_args:
        cmd = f"{cmd} {' '.join(extra_args)}".strip()

    enable_ssh = bool(ssh or ssh_password or ssh_pubkey)
    if enable_ssh:
        cmd = f"{_SSH_PROLOGUE}\n{cmd}"
        env.append({"name": "WUJI_SSH_PASSWORD", "value": ssh_password or ""})
        env.append({"name": "WUJI_SSH_PUBKEY", "value": ssh_pubkey or ""})
        env.append({"name": "WUJI_SSH_PORT", "value": str(ssh_port)})

    # 多机分布式:自动配 rendezvous(MASTER_ADDR/NNODES/NODE_RANK),依赖下方 spec.plugins。
    if nodes > 1:
        cmd = f"{_DIST_PROLOGUE}\n{cmd}"

    container: Dict[str, Any] = {
        "name": "worker",
        "image": image,
        "command": [shell, "-lc" if shell == "bash" else "-c", cmd],
        "workingDir": workspace_path,
        "resources": {
            "limits": {
                "nvidia.com/gpu": str(gpus),
                "cpu": str(cpu),
                "memory": str(memory),
            },
        },
        "volumeMounts": [
            {"name": "ws", "mountPath": workspace_path},
            {"name": "dshm", "mountPath": DSHM_MOUNT},
        ],
    }
    if env:
        container["env"] = env
    port_list = list(ports or [])
    if enable_ssh and ssh_port not in port_list:
        port_list.append(ssh_port)
    if port_list:
        container["ports"] = [
            (
                {"containerPort": p, "name": "ssh"}
                if enable_ssh and p == ssh_port
                else {"containerPort": p}
            )
            for p in port_list
        ]

    volumes: List[Dict[str, Any]] = [
        {"name": "ws", "persistentVolumeClaim": {"claimName": f"{team}-nas"}},
        {
            "name": "dshm",
            "emptyDir": {"medium": "Memory", "sizeLimit": dshm_size},
        },
    ]

    # 只有用到 --shared-data 时才挂共享只读数据集 PVC(否则该 PVC 不必存在)
    if shared_data:
        container["volumeMounts"].insert(
            1, {"name": "datasets", "mountPath": DATASETS_MOUNT, "readOnly": True}
        )
        volumes.append(
            {
                "name": "datasets",
                "persistentVolumeClaim": {"claimName": shared_dataset_pvc, "readOnly": True},
            }
        )

    pod_spec: Dict[str, Any] = {
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": volumes,
    }
    if node:
        pod_spec["nodeSelector"] = {"kubernetes.io/hostname": node}

    spec: Dict[str, Any] = {
        "schedulerName": "volcano",
        "queue": queue,
        "minAvailable": nodes,
        "tasks": [
            {
                "name": "worker",
                "replicas": nodes,
                "template": {
                    "metadata": {"labels": {"wuji.io/job": name}},
                    "spec": pod_spec,
                },
            }
        ],
    }
    # 多机:开 Volcano svc 插件(headless service + VC_WORKER_HOSTS/NUM 供发现)
    # 和 env 插件(VC_TASK_INDEX 作 node rank)。单机不需要,省掉额外的 service/configmap。
    if nodes > 1:
        spec["plugins"] = {"svc": [], "env": []}

    return {
        "apiVersion": "batch.volcano.sh/v1alpha1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": team,
            "labels": {
                "app.kubernetes.io/managed-by": "wuji",
                **({"wuji.io/kind": kind} if kind else {}),
            },
        },
        "spec": spec,
    }
