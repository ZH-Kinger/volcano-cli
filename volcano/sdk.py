"""SDK functions — the programmatic surface of volcano.

Every function here is import-safe (``import volcano; volcano.submit(...)``) and is
also what ``volcano.cli`` calls under the hood. Each resolves the namespace,
loads clients, and turns Kubernetes ``ApiException``\\ s into :class:`WujiError`.
"""

from __future__ import annotations

import base64
import json
import re
import secrets
import time
from typing import Any, Dict, List, Optional, Union

from kubernetes import client
from kubernetes.client.exceptions import ApiException

from .job import build_volcano_job, ssh_secret_name_for_job
from .kube import (
    DEFAULT_ACR_REGISTRY,
    DEFAULT_ACR_REPO_PREFIX,
    DEFAULT_APISERVER,
    GATEKEEPER_ASSIGN_PLURAL,
    GATEKEEPER_ASSIGN_VERSION,
    GATEKEEPER_MUTATIONS_GROUP,
    IMAGE_INDEX_NAME,
    IMAGE_INDEX_NAMESPACE,
    JOB_NAME_LABEL,
    NAS_BASE,
    NAS_SERVER,
    NODE_PUBLIC_IP_ANNOTATION,
    SAVE_REQUEST_LABEL,
    TEAM_LABEL,
    VOLCANO_JOB_GROUP,
    VOLCANO_JOB_PLURAL,
    VOLCANO_JOB_VERSION,
    VOLCANO_QUEUE_GROUP,
    VOLCANO_QUEUE_PLURAL,
    VOLCANO_QUEUE_VERSION,
    WujiError,
    current_namespace,
    humanize_api_exception,
    load_clients,
)

# Pods carry this label (set by build_volcano_job) — the NodePort SSH Service
# selects on it, and it is stable across the job's pods.
POD_JOB_LABEL = "wuji.io/job"

Command = Union[str, List[str]]


def submit(
    *,
    name: str,
    team: Optional[str] = None,
    queue: str = "default",
    image: str,
    command: Command,
    gpus: int = 8,
    nodes: int = 1,
    data: Optional[str] = None,
    shared_data: Optional[str] = None,
    cpu: str = "16",
    memory: str = "64Gi",
    shell: str = "bash",
    workspace_path: str = "/workspace",
    shared_dataset_pvc: str = "shared-datasets",
    ssh: bool = False,
    ssh_password: Optional[str] = None,
    ssh_port: int = 22,
    ports: Optional[List[int]] = None,
    node: Optional[str] = None,
    kind: Optional[str] = None,
    mounts: Optional[List[str]] = None,
    expose_ssh: bool = False,
    ssh_node_port: Optional[int] = None,
) -> Dict[str, Any]:
    """Build and create a Volcano Job on the cluster.

    Args:
        name: job name. team: namespace (defaults to the resolved namespace).
        queue: Volcano queue. image: full ACR image ref. command: training cmd.
        gpus/nodes/data/shared_data/cpu/memory/shared_dataset_pvc: see
            :func:`volcano.job.build_volcano_job`.
        expose_ssh: also create a NodePort Service exposing the pod's sshd on a
            node's elastic public IP (implies ssh). ssh_node_port: pin a specific
            NodePort (30000-32767) instead of letting Kubernetes auto-assign.

    Returns:
        The created object as returned by the API server. When ``expose_ssh``
        is set, the dict carries an extra ``"_ssh_node_port"`` key with the
        allocated NodePort.

    Raises:
        WujiError: on any Kubernetes API failure (message is user-friendly), or
            if SSH is enabled with no ``ssh_password`` given — password-only
            login by design (see punchlist), so callers (the CLI) must resolve
            a password interactively before calling this; submit() itself
            never prompts or invents one (must stay safe to call from scripts).
    """
    ns = current_namespace(team)
    if expose_ssh:
        ssh = True  # exposing SSH is meaningless unless sshd is started
    # 保证:开了 SSH 但没给密码时,总得选一种登录方式,否则 sshd 前置脚本不会启动
    # (它要求 WUJI_SSH_PASSWORD 非空),而 NodePort 仍会建 → 一个对公网开放却无人监听
    # 的死端口。密码登录是唯一方式(不读/不存任何本机凭据,也不自动生成密码);
    # submit() 本身绝不交互问密码(要保持脚本可安全调用),没给密码就直接报错让
    # 调用方(CLI)自己去交互式问用户要。
    if ssh and not ssh_password:
        raise WujiError(
            "开了 SSH 但没给密码:直接调用 submit() 时请传入 ssh_password;"
            "用 volcano dev/train 的 --ssh 会自动交互式问你要一个,不需要自己传。"
        )

    core, _, custom = load_clients()

    # SSH 密码一律经 Secret 的 secretKeyRef 注入,不写 pod 的字面量 env——否则任何
    # 有 pods/jobs 读权限的人(目前是全体注册用户,见 volcano-cluster-viewer)都能读到
    # 明文密码,叠加 --expose-ssh 默认公网映射就是直接的跨租户 root 提权。Secret 必须先
    # 于 Job 建好(pod 起来时就要能解析 secretKeyRef);ownerReference 待 Job 创建成功、
    # 拿到真实 uid 后再补上,kill() 也按 label 做同步兜底删除(见下方)。
    secret_name = None
    if ssh_password:
        secret_name = ssh_secret_name_for_job(name)
        secret_data = {"password": ssh_password}
        secret_body = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": secret_name,
                "namespace": ns,
                "labels": {"app.kubernetes.io/managed-by": "wuji", POD_JOB_LABEL: name},
            },
            "type": "Opaque",
            "stringData": secret_data,
        }
        try:
            core.create_namespaced_secret(namespace=ns, body=secret_body)
        except ApiException as exc:
            if exc.status != 409:
                raise WujiError(humanize_api_exception(exc)) from exc
            # 409:多半是上次提交同名 job 失败后留下的残留 Secret,刷新内容后继续复用。
            try:
                core.patch_namespaced_secret(
                    name=secret_name, namespace=ns, body={"stringData": secret_data}
                )
            except ApiException as exc2:
                raise WujiError(humanize_api_exception(exc2)) from exc2

    manifest = build_volcano_job(
        name=name,
        team=ns,
        queue=queue,
        image=image,
        gpus=gpus,
        nodes=nodes,
        command=command,
        data=data,
        shared_data=shared_data,
        cpu=cpu,
        memory=memory,
        shell=shell,
        workspace_path=workspace_path,
        shared_dataset_pvc=shared_dataset_pvc,
        ssh=ssh,
        ssh_password=ssh_password,
        ssh_port=ssh_port,
        ssh_secret_name=secret_name,
        ports=ports,
        node=node,
        kind=kind,
        mounts=mounts,
    )
    try:
        created = custom.create_namespaced_custom_object(
            group=VOLCANO_JOB_GROUP,
            version=VOLCANO_JOB_VERSION,
            namespace=ns,
            plural=VOLCANO_JOB_PLURAL,
            body=manifest,
        )
    except ApiException as exc:
        if secret_name:  # 别留一个没有对应 job 的密码 Secret
            try:
                core.delete_namespaced_secret(name=secret_name, namespace=ns)
            except ApiException:
                pass
        raise WujiError(humanize_api_exception(exc)) from exc

    if secret_name:
        owner_refs = _job_owner_ref(created, name)
        if owner_refs:
            try:
                core.patch_namespaced_secret(
                    name=secret_name,
                    namespace=ns,
                    body={"metadata": {"ownerReferences": owner_refs}},
                )
            except ApiException:
                pass  # best-effort — kill() still cleans it up by label

    if expose_ssh:
        if nodes > 1:
            # 每个 worker 一个独立公网入口(无 selector NodePort + 精确 Endpoints)
            worker_nps = _create_worker_ssh_services(
                name, ns=ns, nodes=nodes, ssh_port=ssh_port, owner=created
            )
            if isinstance(created, dict):
                created["_ssh_worker_node_ports"] = worker_nps
        else:
            allocated = _create_ssh_service(
                name, ns=ns, ssh_port=ssh_port, node_port=ssh_node_port, owner=created
            )
            if isinstance(created, dict):
                created["_ssh_node_port"] = allocated
    return created


def _create_ssh_service(
    name: str,
    *,
    ns: str,
    ssh_port: int = 22,
    node_port: Optional[int] = None,
    owner: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Create a NodePort Service exposing a job's sshd; return the NodePort.

    The Service selects the job's pods via the ``wuji.io/job`` label and maps a
    node port to the in-container ssh port. If ``node_port`` is given it is
    pinned, otherwise Kubernetes auto-assigns one from 30000-32767.

    When ``owner`` (the created Job object) is provided, the Service gets an
    ownerReference to it so Kubernetes garbage-collects the Service when the Job
    is deleted — not only via ``volcano kill`` but also on natural completion/TTL.
    """
    core, _, _ = load_clients()
    port: Dict[str, Any] = {
        "name": "ssh",
        "port": ssh_port,
        "targetPort": ssh_port,
        "protocol": "TCP",
    }
    if node_port:
        port["nodePort"] = node_port
    metadata: Dict[str, Any] = {
        "name": f"{name}-ssh",
        "namespace": ns,
        "labels": {"app.kubernetes.io/managed-by": "wuji", POD_JOB_LABEL: name},
    }
    owner_uid = None
    if isinstance(owner, dict):
        owner_uid = owner.get("metadata", {}).get("uid")
    if owner_uid:
        metadata["ownerReferences"] = [
            {
                "apiVersion": f"{VOLCANO_JOB_GROUP}/{VOLCANO_JOB_VERSION}",
                "kind": "Job",
                "name": name,
                "uid": owner_uid,
                "blockOwnerDeletion": False,
                "controller": False,
            }
        ]
    body = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": metadata,
        "spec": {
            "type": "NodePort",
            "selector": {POD_JOB_LABEL: name},
            "ports": [port],
        },
    }
    try:
        svc = core.create_namespaced_service(namespace=ns, body=body)
    except ApiException as exc:
        raise WujiError(
            f"任务 {name} 已建并在运行,但暴露 SSH 的 NodePort Service 创建失败:"
            f"{humanize_api_exception(exc)}。如不需要该任务,请执行: volcano kill {name}"
        ) from exc
    ports = svc.spec.ports if svc.spec else None
    return ports[0].node_port if ports else None


def ssh_endpoint(name: str, *, team: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve a job's public SSH endpoint (for --expose-ssh jobs).

    Returns:
        ``{node, public_ip, node_port}`` if a ``<name>-ssh`` NodePort Service
        exists (public_ip/node may be None until a pod is scheduled), else None.
    """
    ns = current_namespace(team)
    core, _, _ = load_clients()
    try:
        svc = core.read_namespaced_service(name=f"{name}-ssh", namespace=ns)
    except ApiException:
        return None
    ports = (svc.spec.ports or []) if svc.spec else []
    node_port = ports[0].node_port if ports else None
    if not node_port:
        return None

    node = None
    try:
        pods = core.list_namespaced_pod(
            namespace=ns, label_selector=f"{JOB_NAME_LABEL}={name}"
        )
        cand = [p for p in pods.items if p.spec and p.spec.node_name]
        running = [
            p for p in cand if p.status and p.status.phase == "Running"
        ]
        if running or cand:
            node = (running or cand)[0].spec.node_name
    except ApiException:
        pass

    public_ip = None
    if node:
        try:
            nd = core.read_node(name=node)
            public_ip = (nd.metadata.annotations or {}).get(
                NODE_PUBLIC_IP_ANNOTATION
            )
        except ApiException:
            pass
    return {"node": node, "public_ip": public_ip, "node_port": node_port}


# --------------------------------------------------------------------------- #
# per-worker public SSH (multi-node --expose-ssh)
# --------------------------------------------------------------------------- #
def _worker_svc_name(name: str, worker: int) -> str:
    """Deterministic per-worker SSH Service/Endpoints name."""
    return f"{name}-ssh-{worker}"


def _job_owner_ref(owner: Optional[Dict[str, Any]], name: str) -> Optional[list]:
    """Build an ownerReference list pointing at the Volcano Job, or None."""
    uid = owner.get("metadata", {}).get("uid") if isinstance(owner, dict) else None
    if not uid:
        return None
    return [
        {
            "apiVersion": f"{VOLCANO_JOB_GROUP}/{VOLCANO_JOB_VERSION}",
            "kind": "Job",
            "name": name,
            "uid": uid,
            "blockOwnerDeletion": False,
            "controller": False,
        }
    ]


def _create_worker_ssh_services(
    name: str, *, ns: str, nodes: int, ssh_port: int, owner: Optional[Dict[str, Any]]
) -> Dict[int, Optional[int]]:
    """Create one selector-less NodePort Service per worker (multi-node expose).

    Selector-less: each Service is paired with a manually-managed Endpoints
    (see :func:`ensure_worker_endpoint`) pointing at exactly that worker's pod IP,
    so ``<node-EIP>:<nodePort_i>`` reaches worker *i* specifically (not round-robin).
    Services get an ownerReference to the Job for GC.

    Returns:
        ``{worker_index: node_port}``.
    """
    core, _, _ = load_clients()
    refs = _job_owner_ref(owner, name)
    out: Dict[int, Optional[int]] = {}
    for i in range(nodes):
        meta: Dict[str, Any] = {
            "name": _worker_svc_name(name, i),
            "namespace": ns,
            "labels": {
                "app.kubernetes.io/managed-by": "wuji",
                POD_JOB_LABEL: name,
                "wuji.io/worker": str(i),
            },
        }
        if refs:
            meta["ownerReferences"] = refs
        body = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": meta,
            "spec": {  # no selector — Endpoints managed manually per worker
                "type": "NodePort",
                "ports": [
                    {"name": "ssh", "port": ssh_port, "targetPort": ssh_port, "protocol": "TCP"}
                ],
            },
        }
        try:
            svc = core.create_namespaced_service(namespace=ns, body=body)
        except ApiException as exc:
            raise WujiError(
                f"任务 {name} 已建并在运行,但 worker {i} 的 SSH Service 创建失败:"
                f"{humanize_api_exception(exc)}。如不需要该任务,请执行: volcano kill {name}"
            ) from exc
        out[i] = svc.spec.ports[0].node_port if svc.spec and svc.spec.ports else None
    return out


def ensure_worker_endpoint(
    name: str, worker: int, *, team: Optional[str] = None, ssh_port: int = 22
) -> Optional[str]:
    """Upsert the Endpoints for a worker's SSH Service to its current pod IP.

    Reconcile-on-access: reads pod ``<name>-worker-<worker>``, and points the
    ``<name>-ssh-<worker>`` Endpoints at ``podIP:ssh_port``. Handles pod restarts
    (IP change) by replacing the Endpoints. The Endpoints is owned by the pod so
    Kubernetes GCs it when the pod goes away.

    Returns:
        The pod IP the Endpoints now targets, or None if the pod has no IP yet.
    """
    ns = current_namespace(team)
    core, _, _ = load_clients()
    pod_name = f"{name}-worker-{worker}"
    try:
        pod = core.read_namespaced_pod(name=pod_name, namespace=ns)
    except ApiException:
        return None
    pod_ip = pod.status.pod_ip if pod.status else None
    if not pod_ip:
        return None
    ep_name = _worker_svc_name(name, worker)
    meta: Dict[str, Any] = {
        "name": ep_name,
        "namespace": ns,
        "labels": {"app.kubernetes.io/managed-by": "wuji", POD_JOB_LABEL: name},
    }
    pod_uid = pod.metadata.uid if pod.metadata else None
    if pod_uid:
        meta["ownerReferences"] = [
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "name": pod_name,
                "uid": pod_uid,
                "blockOwnerDeletion": False,
                "controller": False,
            }
        ]
    body = {
        "apiVersion": "v1",
        "kind": "Endpoints",
        "metadata": meta,
        "subsets": [
            {
                "addresses": [{"ip": pod_ip}],
                "ports": [{"name": "ssh", "port": ssh_port, "protocol": "TCP"}],
            }
        ],
    }
    try:
        core.replace_namespaced_endpoints(name=ep_name, namespace=ns, body=body)
    except ApiException:
        try:
            core.create_namespaced_endpoints(namespace=ns, body=body)
        except ApiException:
            return None
    return pod_ip


def worker_ssh_endpoint(
    name: str, worker: int, *, team: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Resolve a specific worker's public SSH endpoint (multi-node expose).

    Reads the ``<name>-ssh-<worker>`` NodePort Service, reconciles its Endpoints
    to the worker pod's current IP, and resolves the worker's node public IP.

    Returns:
        ``{worker, pod, node, public_ip, node_port}`` or None if no such Service.
    """
    ns = current_namespace(team)
    core, _, _ = load_clients()
    try:
        svc = core.read_namespaced_service(
            name=_worker_svc_name(name, worker), namespace=ns
        )
    except ApiException:
        return None
    ports = (svc.spec.ports or []) if svc.spec else []
    if not ports:
        return None
    node_port = ports[0].node_port
    target_port = ports[0].target_port or 22
    try:
        ensure_worker_endpoint(name, worker, team=team, ssh_port=int(target_port))
    except (ApiException, ValueError, TypeError):
        pass
    pod_name = f"{name}-worker-{worker}"
    node = public_ip = None
    try:
        pod = core.read_namespaced_pod(name=pod_name, namespace=ns)
        node = pod.spec.node_name if pod.spec else None
    except ApiException:
        pass
    if node:
        try:
            nd = core.read_node(name=node)
            public_ip = (nd.metadata.annotations or {}).get(NODE_PUBLIC_IP_ANNOTATION)
        except ApiException:
            pass
    return {
        "worker": worker,
        "pod": pod_name,
        "node": node,
        "public_ip": public_ip,
        "node_port": node_port,
    }


# --------------------------------------------------------------------------- #
# volcano save — platform-delegated commit of a live container to ACR
# --------------------------------------------------------------------------- #
def _resolve_image_ref(
    tag: str,
    *,
    team: str,
    registry: str = DEFAULT_ACR_REGISTRY,
    repo_prefix: str = DEFAULT_ACR_REPO_PREFIX,
) -> str:
    """Build the full ACR ref a saved image lands at — **team-scoped, mandatory**.

    Saved images always go under ``<registry>/<repo_prefix>/<team>/`` so a user
    can never overwrite a base image (``<repo_prefix>/<base>``) or another team's
    tag. The ``wuji-saver`` enforces the identical rule server-side from the
    request's *own* namespace; this function is only the client-side prediction
    of that ref (for display).

    ``tag`` must be a short ``repo[:tag]`` with **no registry host** — a full
    registry ref is rejected, so pushing to an arbitrary repo isn't expressible.
    A missing ``:version`` defaults to ``:latest``.
    """
    tag = tag.strip().strip("/")
    if not tag:
        raise WujiError("镜像 tag 不能为空,例如: --tag myenv:v1")
    if not team:
        raise WujiError("保存镜像需要确定团队(namespace)。")
    first = tag.split("/", 1)[0]
    looks_like_host = "." in first or ":" in first or first == "localhost"
    if "/" in tag and looks_like_host:
        raise WujiError(
            "--tag 请用短名(如 myenv:v1),不要带 registry 地址;"
            "镜像会存到你团队专属的 ACR 空间。"
        )
    ref = f"{registry}/{repo_prefix}/{team}/{tag}"
    # ensure a version is present on the final path component
    last = ref.rsplit("/", 1)[-1]
    if ":" not in last:
        ref = f"{ref}:latest"
    return ref


def save_image(
    name: str,
    *,
    tag: str,
    team: Optional[str] = None,
    worker: int = 0,
    container: str = "worker",
    registry: str = DEFAULT_ACR_REGISTRY,
    repo_prefix: str = DEFAULT_ACR_REPO_PREFIX,
    wait: bool = True,
    timeout: int = 1800,
    poll_interval: int = 5,
) -> Dict[str, Any]:
    """Ask the platform to commit a running dev/train container and push it to ACR.

    This creates a *SaveRequest* ConfigMap (labelled :data:`SAVE_REQUEST_LABEL`)
    in the team namespace describing the target pod/container and the desired
    image ref. The privileged ``wuji-saver`` DaemonSet — the only component that
    holds the containerd socket and the ACR **write** credential — picks it up on
    the pod's node, runs ``nerdctl -n k8s.io commit`` + ``push``, and writes the
    outcome back into the ConfigMap's ``status`` field. The caller (user) needs
    no privilege and no registry credential.

    Args:
        name: the dev/train job name whose container to snapshot.
        tag: short image name (``myenv:v1``). It is always scoped to the team —
            the final ref is ``<registry>/<repo_prefix>/<team>/<tag>`` — so it
            can't overwrite a base image or another team's tag. A ``--tag`` that
            includes a registry host is rejected.
        team: namespace override. worker: worker ordinal (pod ``<name>-worker-<n>``).
        container: container name inside the pod (Volcano jobs use ``worker``).
        wait: poll until the saver finishes (or ``timeout`` seconds elapse).
        poll_interval: seconds between status polls.

    Returns:
        ``{request, image, status, node, message}``. When ``wait`` is False the
        status is ``"Pending"`` and node/message may be empty.

    Raises:
        WujiError: on API failure, if the target pod is missing, or on timeout
            (the message includes the request name so the user can check later).
    """
    ns = current_namespace(team)
    short_tag = tag.strip().strip("/")
    image = _resolve_image_ref(
        short_tag, team=ns, registry=registry, repo_prefix=repo_prefix
    )
    core, _, _ = load_clients()

    pod_name = f"{name}-worker-{worker}"
    # Fail fast with a clear message rather than leaving a request the saver will
    # never satisfy (pod not there / not scheduled yet).
    try:
        pod = core.read_namespaced_pod(name=pod_name, namespace=ns)
    except ApiException as exc:
        if exc.status == 404:
            raise WujiError(
                f"找不到容器 {pod_name}(ns={ns})。确认任务名/worker 正确,且它正在 Running。"
            ) from exc
        raise WujiError(humanize_api_exception(exc)) from exc
    phase = (pod.status.phase if pod.status else None) or "Unknown"
    if phase != "Running":
        raise WujiError(
            f"容器 {pod_name} 当前是 {phase},不是 Running,无法提交。等它 Running 后再试。"
        )

    req_name = f"wuji-save-{name}-w{worker}-{secrets.token_hex(3)}"
    body = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": req_name,
            "namespace": ns,
            "labels": {
                "app.kubernetes.io/managed-by": "wuji",
                SAVE_REQUEST_LABEL: "true",
                POD_JOB_LABEL: name,
            },
        },
        "data": {
            "pod": pod_name,
            "namespace": ns,
            "container": container,
            # saver rebuilds the authoritative ref from the request's OWN ns +
            # this short tag (it ignores 'image' below, which is only the
            # client-side prediction shown to the user / used as index fallback).
            "tag": short_tag,
            "image": image,
            "status": "Pending",
            "requestedBy": ns,
        },
    }
    try:
        core.create_namespaced_config_map(namespace=ns, body=body)
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc

    result = {
        "request": req_name,
        "image": image,
        "status": "Pending",
        "node": "",
        "message": "",
    }
    if not wait:
        return result

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll_interval)
        try:
            cm = core.read_namespaced_config_map(name=req_name, namespace=ns)
        except ApiException:
            continue  # transient; keep polling until deadline
        data = cm.data or {}
        status = data.get("status", "Pending")
        result.update(
            status=status,
            node=data.get("node", ""),
            message=data.get("message", ""),
        )
        if status in ("Succeeded", "Failed"):
            if status == "Failed":
                raise WujiError(
                    f"保存镜像失败: {data.get('message') or '(saver 未给出原因)'}"
                    f"(请求 {req_name},可 kubectl -n {ns} get cm {req_name} -o yaml 查看)"
                )
            return result
    raise WujiError(
        f"等待保存超时({timeout}s)。saver 可能未在节点运行,或提交/推送较慢。"
        f"请求仍在: {req_name}(稍后 kubectl -n {ns} get cm {req_name} -o jsonpath='{{.data.status}}')"
    )


def list_images(
    *,
    index_namespace: str = IMAGE_INDEX_NAMESPACE,
    index_name: str = IMAGE_INDEX_NAME,
    team: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List images users can pull, from the platform image index ConfigMap.

    The index is populated two ways: ``wuji-saver`` appends every image produced
    by ``volcano save``, and admins seed base images. Reading it needs no ACR
    credential (the ACR pull token cannot list the catalog anyway).

    Each ConfigMap ``data`` value is a JSON object; malformed/legacy entries fall
    back to treating the raw value as the image ref.

    Args:
        index_namespace/index_name: where the index ConfigMap lives.
        team: if given, only return entries whose ``team`` matches (base images,
            which have no team, are always included).

    Returns:
        ``[{image, kind, team, node, created}]`` sorted by (kind, image); empty
        list if the index doesn't exist yet.
    """
    core, _, _ = load_clients()
    try:
        cm = core.read_namespaced_config_map(name=index_name, namespace=index_namespace)
    except ApiException as exc:
        if exc.status == 404:
            return []  # index not created yet → nothing to show, not an error
        raise WujiError(humanize_api_exception(exc)) from exc

    out: List[Dict[str, Any]] = []
    for _key, raw in (cm.data or {}).items():
        try:
            entry = json.loads(raw)
            if not isinstance(entry, dict):
                raise ValueError
        except (ValueError, TypeError):
            entry = {"image": raw}
        image = entry.get("image")
        if not image:
            continue
        row = {
            "image": image,
            "kind": entry.get("kind", "saved"),
            "team": entry.get("team", ""),
            "node": entry.get("node", ""),
            "created": entry.get("created", entry.get("ts", "")),
        }
        if team and row["team"] and row["team"] != team:
            continue
        out.append(row)
    # str() guard: a hand-seeded base entry with a non-string image must not
    # crash the whole `volcano images` via a str/int comparison in the sort key.
    out.sort(key=lambda r: (str(r["kind"]), str(r["image"])))
    return out


def list_jobs(
    *, team: Optional[str] = None, kind: Optional[str] = None,
    all_namespaces: bool = False,
) -> List[Dict[str, Any]]:
    """List Volcano Jobs in the team namespace (or all namespaces).

    Args:
        team: namespace override.
        kind: if given ("dev"/"train"), only return jobs with that
            ``wuji.io/kind`` label. Jobs with no kind label match neither
            filter (they are only shown by an unfiltered list).
        all_namespaces: if True, list across every namespace — genuinely
            admin-only (see :func:`_require_admin`): a Job's pod template
            carries full env/command, same sensitivity as a raw Pod, so this
            must not be handed to every registered user the way per-namespace
            listing is.

    Returns:
        A list of ``{namespace, name, phase, queue, gpus, kind, age}`` summary dicts.

    Raises:
        WujiError: if ``all_namespaces`` is set and the caller isn't an admin,
            or on any API failure.
    """
    _, _, custom = load_clients()
    try:
        if all_namespaces:
            _require_admin()
            resp = custom.list_cluster_custom_object(
                group=VOLCANO_JOB_GROUP,
                version=VOLCANO_JOB_VERSION,
                plural=VOLCANO_JOB_PLURAL,
            )
        else:
            ns = current_namespace(team)
            resp = custom.list_namespaced_custom_object(
                group=VOLCANO_JOB_GROUP,
                version=VOLCANO_JOB_VERSION,
                namespace=ns,
                plural=VOLCANO_JOB_PLURAL,
            )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc

    out: List[Dict[str, Any]] = []
    for item in resp.get("items", []):
        meta = item.get("metadata", {})
        spec = item.get("spec", {})
        tasks = spec.get("tasks", [{}])
        gpus = _job_gpus(tasks)
        job_kind = (meta.get("labels", {}) or {}).get("wuji.io/kind", "-")
        if kind and job_kind != kind:
            continue
        out.append(
            {
                "namespace": meta.get("namespace", "-"),
                "name": meta.get("name", "?"),
                "phase": item.get("status", {}).get("state", {}).get("phase", "Unknown"),
                "queue": spec.get("queue", "-"),
                "gpus": gpus,
                "kind": job_kind,
                "age": meta.get("creationTimestamp", "-"),
            }
        )
    return out


def status(name: str, *, team: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a Volcano Job's phase and its pods.

    Args:
        name: job name. team: namespace override.

    Returns:
        ``{name, phase, message, pods: [{name, phase, node}]}``.
    """
    ns = current_namespace(team)
    core, _, custom = load_clients()
    try:
        job = custom.get_namespaced_custom_object(
            group=VOLCANO_JOB_GROUP,
            version=VOLCANO_JOB_VERSION,
            namespace=ns,
            plural=VOLCANO_JOB_PLURAL,
            name=name,
        )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc

    state = job.get("status", {}).get("state", {})
    pods: List[Dict[str, str]] = []
    gpu = 0
    cpu_m = 0.0
    mem_gi = 0.0
    try:
        pod_list = core.list_namespaced_pod(
            namespace=ns, label_selector=f"{JOB_NAME_LABEL}={name}"
        )
        for pod in pod_list.items:
            pods.append(
                {
                    "name": pod.metadata.name,
                    "phase": (pod.status.phase or "Unknown") if pod.status else "Unknown",
                    "node": (pod.spec.node_name or "-") if pod.spec else "-",
                }
            )
            containers = (pod.spec.containers or []) if pod.spec else []
            for c in containers:
                res = c.resources
                req = (res.requests or {}) if res else {}
                lim = (res.limits or {}) if res else {}
                gpu += _gpu(lim.get("nvidia.com/gpu"))
                cpu_m += _cpu_to_m(req.get("cpu"))
                mem_gi += _mem_to_gi(req.get("memory"))
    except ApiException:
        pass  # pods are best-effort; job status is the primary answer

    return {
        "name": name,
        "phase": state.get("phase", "Unknown"),
        "message": state.get("message", ""),
        "pods": pods,
        "gpu": gpu,
        "cpu_m": cpu_m,
        "mem_gi": mem_gi,
    }


def kill(name: str, *, team: Optional[str] = None) -> None:
    """Delete a Volcano Job.

    Args:
        name: job name. team: namespace override.

    Raises:
        WujiError: on API failure.
    """
    ns = current_namespace(team)
    core, _, custom = load_clients()
    try:
        custom.delete_namespaced_custom_object(
            group=VOLCANO_JOB_GROUP,
            version=VOLCANO_JOB_VERSION,
            namespace=ns,
            plural=VOLCANO_JOB_PLURAL,
            name=name,
        )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc

    # best-effort: clean up any --expose-ssh NodePort Services (single `<name>-ssh`
    # and per-worker `<name>-ssh-<i>`), matched by the job label. Their Endpoints
    # are pod-owned and GC'd when the pods go away.
    try:
        svcs = core.list_namespaced_service(
            namespace=ns, label_selector=f"{POD_JOB_LABEL}={name}"
        )
        for s in svcs.items:
            try:
                core.delete_namespaced_service(name=s.metadata.name, namespace=ns)
            except ApiException:
                pass
    except ApiException:
        pass

    # best-effort: clean up the SSH password Secret (see submit()). It also
    # carries an ownerReference to the Job for GC on natural deletion/TTL, but that's
    # async — this synchronous, label-matched delete is the immediate backstop
    # (mirrors the Service cleanup above).
    try:
        secs = core.list_namespaced_secret(
            namespace=ns, label_selector=f"{POD_JOB_LABEL}={name}"
        )
        for s in secs.items:
            try:
                core.delete_namespaced_secret(name=s.metadata.name, namespace=ns)
            except ApiException:
                pass
    except ApiException:
        pass


def _first_pod(core, ns: str, name: str) -> Optional[str]:
    """Return the name of the first pod belonging to a job, or None."""
    pods = core.list_namespaced_pod(
        namespace=ns, label_selector=f"{JOB_NAME_LABEL}={name}"
    )
    if not pods.items:
        return None
    return pods.items[0].metadata.name


def logs(
    name: str,
    *,
    team: Optional[str] = None,
    follow: bool = False,
    tail: int = 200,
    pod: Optional[str] = None,
    worker: Optional[int] = None,
):
    """Return (or stream) logs of one of a job's pods.

    Args:
        name: job name. team: namespace override.
        follow: if True, return a line iterator that streams live logs.
        tail: number of trailing lines to fetch (non-follow mode).
        pod: explicit pod name to read (wins over ``worker``).
        worker: worker ordinal — resolves to pod ``<name>-worker-<n>`` (Volcano's
            deterministic naming). Defaults to the first pod when neither given.

    Returns:
        A ``str`` of log text when ``follow`` is False, otherwise a line
        generator you can iterate over.

    Raises:
        WujiError: if the job has no pods yet, or on API failure.
    """
    ns = current_namespace(team)
    core, _, _ = load_clients()
    try:
        if pod:
            target = pod
        elif worker is not None:
            target = f"{name}-worker-{worker}"
        else:
            target = _first_pod(core, ns, name)
        pod = target
        if not pod:
            raise WujiError(
                f"任务 {name} 还没有 Pod(可能仍在排队/未调度)。稍后再试。"
            )
        if follow:
            from kubernetes import watch

            w = watch.Watch()
            return w.stream(
                core.read_namespaced_pod_log,
                name=pod,
                namespace=ns,
                follow=True,
            )
        return core.read_namespaced_pod_log(
            name=pod, namespace=ns, tail_lines=tail
        )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc


def list_pvcs(*, team: Optional[str] = None) -> List[Dict[str, Any]]:
    """List PVCs in the team namespace (mountable volumes).

    Returns a list of ``{name, status, capacity, storageclass, volume}`` dicts.
    """
    ns = current_namespace(team)
    core, _, _ = load_clients()
    try:
        resp = core.list_namespaced_persistent_volume_claim(namespace=ns)
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc
    out: List[Dict[str, Any]] = []
    for item in resp.items:
        st = item.status
        cap = ""
        if st and getattr(st, "capacity", None):
            cap = (st.capacity or {}).get("storage", "")
        out.append(
            {
                "name": item.metadata.name,
                "status": (st.phase if st else "?") or "?",
                "capacity": cap or "-",
                "storageclass": item.spec.storage_class_name or "-",
                "volume": item.spec.volume_name or "-",
            }
        )
    return out


def list_queues() -> List[Dict[str, Any]]:
    """List Volcano Queues with GPU/CPU/memory deserved/capability/allocated.

    Returns:
        A list of ``{name, state, deserved, capability, allocated}`` dicts, where
        each of deserved/capability/allocated is ``{gpu, cpu_m, mem_gi}``.
    """
    _, _, custom = load_clients()
    try:
        resp = custom.list_cluster_custom_object(
            group=VOLCANO_QUEUE_GROUP,
            version=VOLCANO_QUEUE_VERSION,
            plural=VOLCANO_QUEUE_PLURAL,
        )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc

    def _res(d: Optional[Dict[str, Any]]) -> Dict[str, float]:
        d = d or {}
        return {
            "gpu": _gpu(d.get("nvidia.com/gpu")),
            "cpu_m": _cpu_to_m(d.get("cpu")),
            "mem_gi": _mem_to_gi(d.get("memory")),
        }

    out: List[Dict[str, Any]] = []
    for item in resp.get("items", []):
        spec = item.get("spec", {})
        st = item.get("status", {})
        out.append(
            {
                "name": item.get("metadata", {}).get("name", "?"),
                "state": st.get("state", "-") or "-",
                "deserved": _res(spec.get("deserved")),
                "capability": _res(spec.get("capability")),
                "allocated": _res(st.get("allocated")),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# resource-quantity parsers + node top
# --------------------------------------------------------------------------- #
_MEM_UNITS = {
    "Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30, "Ti": 2 ** 40,
    "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
}


def _gpu(v: Any) -> int:
    """Parse an ``nvidia.com/gpu`` quantity to int (0 on failure)."""
    try:
        return int(v or 0)
    except (ValueError, TypeError):
        return 0


def _cpu_to_m(v: Any) -> float:
    """Parse a k8s CPU quantity to millicores (e.g. ``"16"``→16000, ``"1232m"``→1232)."""
    if v is None:
        return 0.0
    s = str(v)
    if s.endswith("m"):
        try:
            return float(s[:-1])
        except ValueError:
            return 0.0
    try:
        return float(s) * 1000.0
    except ValueError:
        return 0.0


def _mem_to_gi(v: Any) -> float:
    """Parse a k8s memory quantity to GiB (handles Ki/Mi/Gi/Ti + plain bytes)."""
    if v is None:
        return 0.0
    s = str(v)
    for unit, mult in _MEM_UNITS.items():
        if s.endswith(unit):
            try:
                return float(s[: -len(unit)]) * mult / (2 ** 30)
            except ValueError:
                return 0.0
    try:
        return float(s) / (2 ** 30)  # plain bytes
    except ValueError:
        return 0.0


# gpu-snapshot 的每节点 ConfigMap(见 gpu-snapshot.yaml/.py)是 top_nodes()/
# usage_by_namespace() 现在唯一的数据源,不再直连集群级 pods API——那条路子需要
# volcano-cluster-viewer 给 pods:get/list/watch,等于把每个 pod 的完整 env/command
# 都暴露给全体用户(见 punchlist §1 的实锤攻击链:SSH 密码曾经就是这么被读到的)。
#
# 权限收窄成只对这 5 个已知 ConfigMap 名字 get(见 gpu-snapshot.yaml 的
# resourceNames),真实验证过 k8s RBAC 的 resourceNames 只对 get 生效——list/watch
# 整个 namespace 会被直接拒绝,没法"按标签一把拉全部"。所以这里是拿 core.list_node()
# 已经允许的节点名单,逐个 get 对应的 gpu-usage-<node>,读不到(尚未发布 / 该节点还
# 没加进 resourceNames)就跳过那一个,不影响其它节点。
#
# 已知取舍(相对旧的直读 pods 实现):
# 1. 只有 Running 且 gpu>0 的 pod 才会出现——gpu-snapshot 采集器本来就只记这类 pod
#    (见 gpu-snapshot.py 的 list_k8s_gpu_pods),Pending(还没分配 node)、纯 cpu/
#    内存不占 GPU 的 pod 看不到。Pending 的可见性可以靠 volcano queue / PodGroup
#    状态弥补,那条本来就是安全的(PodGroup 不含 pod 模板,没有这个泄露问题)。
# 2. 数据不是实时的,有 gpu-snapshot 的采集间隔(默认 5s)。
_GPU_SNAPSHOT_NS = "wuji-system"
_GPU_SNAPSHOT_CM_PREFIX = "gpu-usage-"


def _list_gpu_nodes(core) -> List[Dict[str, Any]]:
    """Cluster nodes exposing ``nvidia.com/gpu`` (raw dicts, fresh every call)."""
    try:
        nresp = core.list_node(_preload_content=False)
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc
    nodes = json.loads(nresp.data).get("items", [])
    return [
        nd for nd in nodes
        if _gpu((nd.get("status", {}).get("allocatable") or {}).get("nvidia.com/gpu")) > 0
    ]


def _gpu_snapshot_entries(core, node_names: List[str]) -> List[Dict[str, Any]]:
    """Read each node's ``gpu-usage-<node>`` ConfigMap and flatten their
    ``k8s_pods.json`` into one list, each entry tagged with its source node.

    One ``get`` per already-known node name — not a ``list`` over the namespace
    (regular users don't have that verb here, see module comment above). A node
    whose ConfigMap can't be read (not published yet, or missing from
    ``gpu-snapshot.yaml``'s ``resourceNames``) is silently skipped.
    """
    out: List[Dict[str, Any]] = []
    for node in node_names:
        try:
            cm = core.read_namespaced_config_map(
                name=f"{_GPU_SNAPSHOT_CM_PREFIX}{node}", namespace=_GPU_SNAPSHOT_NS
            )
        except ApiException:
            continue
        try:
            pods = json.loads((cm.data or {}).get("k8s_pods.json", "[]") or "[]")
        except (TypeError, ValueError):
            pods = []
        for p in pods:
            p.setdefault("node", node)
            out.append(p)
    return out


def top_nodes() -> List[Dict[str, Any]]:
    """Per GPU-node resource usage: GPU/CPU/memory allocated vs capacity.

    "allocated" = sum of GPU limits and CPU/memory requests for Running pods
    on that node, as recorded by the gpu-snapshot DaemonSet (see module
    comment above for the data source and its tradeoffs vs. a live pods read).

    Returns:
        A list (sorted by node name) of dicts with keys ``node``,
        ``gpu_used/gpu_cap``, ``cpu_used_m/cpu_cap_m``, ``mem_used_gi/mem_cap_gi``.
        Only nodes that expose ``nvidia.com/gpu`` are included.
    """
    core, _, _ = load_clients()
    gpu_nodes = _list_gpu_nodes(core)
    names = [nd.get("metadata", {}).get("name", "?") for nd in gpu_nodes]

    used: Dict[str, Dict[str, float]] = {}
    for p in _gpu_snapshot_entries(core, names):
        acc = used.setdefault(p.get("node", ""), {"gpu": 0, "cpu_m": 0.0, "mem_gi": 0.0})
        acc["gpu"] += int(p.get("gpus", 0) or 0)
        acc["cpu_m"] += float(p.get("cpu_m", 0) or 0)
        acc["mem_gi"] += float(p.get("mem_gi", 0) or 0)

    out: List[Dict[str, Any]] = []
    for nd in gpu_nodes:
        alloc = nd.get("status", {}).get("allocatable") or {}
        name = nd.get("metadata", {}).get("name", "?")
        u = used.get(name, {"gpu": 0, "cpu_m": 0.0, "mem_gi": 0.0})
        out.append(
            {
                "node": name,
                "gpu_used": int(u["gpu"]), "gpu_cap": _gpu(alloc.get("nvidia.com/gpu")),
                "cpu_used_m": u["cpu_m"], "cpu_cap_m": _cpu_to_m(alloc.get("cpu")),
                "mem_used_gi": u["mem_gi"], "mem_cap_gi": _mem_to_gi(alloc.get("memory")),
            }
        )
    out.sort(key=lambda x: x["node"])
    return out


def usage_by_namespace(team: Optional[str] = None) -> List[Dict[str, Any]]:
    """按 namespace(=人/团队)聚合 gpu-snapshot 记录的 GPU/CPU/内存用量。

    数据来源/取舍见本文件上方 gpu-snapshot 相关注释——只统计 Running 且 gpu>0
    的 pod,Pending 与纯 cpu/内存(不占 GPU)的 pod 不在此列;``team=<ns>`` 因此
    是"这个 ns 正在占用的 GPU 相关 pod",不是这个 ns 的全部资源占用。

    Args:
        team: 只看某个 namespace;为 None 时返回所有"在占 GPU"的 namespace。

    Returns:
        ``[{namespace, gpu, cpu_m, mem_gi, pods}]``,按 GPU 降序。
    """
    core, _, _ = load_clients()
    gpu_nodes = _list_gpu_nodes(core)
    names = [nd.get("metadata", {}).get("name", "?") for nd in gpu_nodes]

    agg: Dict[str, Dict[str, float]] = {}
    for p in _gpu_snapshot_entries(core, names):
        ns = p.get("ns", "")
        a = agg.setdefault(ns, {"gpu": 0, "cpu_m": 0.0, "mem_gi": 0.0, "pods": 0})
        a["pods"] += 1
        a["gpu"] += int(p.get("gpus", 0) or 0)
        a["cpu_m"] += float(p.get("cpu_m", 0) or 0)
        a["mem_gi"] += float(p.get("mem_gi", 0) or 0)

    rows = [{"namespace": ns, **v} for ns, v in agg.items()]
    if team:
        rows = [r for r in rows if r["namespace"] == team]
    else:
        rows = [r for r in rows if r["gpu"] > 0]  # 默认只看在占 GPU 的
    rows.sort(key=lambda r: (-r["gpu"], -r["cpu_m"]))
    return rows


def _job_gpus(tasks: List[Dict[str, Any]]) -> int:
    """Sum GPU limits across a Volcano Job's tasks (replicas × per-pod gpus)."""
    total = 0
    for task in tasks:
        replicas = int(task.get("replicas", 1) or 1)
        containers = (
            task.get("template", {}).get("spec", {}).get("containers", [])
        )
        per_pod = 0
        for c in containers:
            lim = c.get("resources", {}).get("limits", {})
            try:
                per_pod += int(lim.get("nvidia.com/gpu", 0) or 0)
            except (ValueError, TypeError):
                pass
        total += replicas * per_pod
    return total


# --------------------------------------------------------------------------- #
# volcano register — admin-only one-shot team onboarding + kubeconfig
# --------------------------------------------------------------------------- #
# 把 platform/add-team.sh + grant-volcano-kubeconfig.sh 的整套开团队逻辑收进 CLI:
# 建 ns + NAS PV/PVC + SA/Role/RoleBinding + 只读 ClusterRole 绑定 + 长期 token +
# 生成团队 kubeconfig。**不碰 ACR**——平台 credential-helper 自动把
# acr-credential-secret-aggregation 注入每个 ns 并绑 default SA,拉镜像天然可用。
# 需要管理员 kubeconfig(能建 namespace / clusterrolebinding),用 SelfSubjectAccessReview 门禁。

_TEAM_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

# 团队 Role 授权面(namespaced);与 grant-volcano-kubeconfig.sh 一致。
_TEAM_ROLE_RULES = [
    {
        "apiGroups": ["batch.volcano.sh"],
        "resources": ["jobs", "jobs/status"],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
    {
        "apiGroups": ["scheduling.volcano.sh"],
        "resources": ["podgroups", "queues"],
        "verbs": ["get", "list", "watch"],
    },
    {
        "apiGroups": [""],
        "resources": ["resourcequotas"],
        "verbs": ["get", "list", "watch"],
    },
    {
        "apiGroups": [""],
        "resources": [
            "pods", "pods/log", "pods/exec", "pods/portforward",
            "services", "endpoints", "configmaps", "persistentvolumeclaims", "events",
        ],
        "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
    },
]

# 共享只读集群视图(volcano top/usage/queue、ls -A、--expose-ssh 公网 IP 解析)。
_CLUSTER_VIEWER_NAME = "volcano-cluster-viewer"
# `pods` used to be here (cluster-wide get/list/watch) so top_nodes()/
# usage_by_namespace() could read every pod's spec directly — that's exactly
# the leak documented in punchlist §1 (full env/command, incl. plaintext SSH
# passwords before the Secret fix, visible to every registered user). Those two
# functions now read the gpu-snapshot ConfigMaps instead (see the comment above
# top_nodes()), so `pods` is dropped from this ClusterRole entirely.
#
# `jobs` (batch.volcano.sh) used to be here too — SAME leak, since a Job's pod
# template carries the same full env/command as a raw Pod (confirmed exploited
# live: a pre-fix job's plaintext WUJI_SSH_PASSWORD was readable cluster-wide
# through exactly this rule). `list_jobs(all_namespaces=True)` (the `--all`
# flag) is the only consumer, and its own docstring already called this an
# "admin view" — admins don't need it here at all, they get real cluster-wide
# jobs CRUD (and everything else) via the built-in `cluster-admin` ClusterRole
# (see grant_admin() below). So `jobs` is dropped from this ClusterRole entirely,
# and `list_jobs()` now calls `_require_admin()` itself when all_namespaces=True
# — regular users keep their own-namespace job listing (via the per-team Role),
# they just lose the accidental cluster-wide `--all` that nothing intended them
# to have.
_CLUSTER_VIEWER_RULES = [
    # namespaces get/list/watch: read-only, lets a user read their own ns's
    # wuji.io/team label so submit can default --queue to their team queue.
    {"apiGroups": [""], "resources": ["nodes", "namespaces"], "verbs": ["get", "list", "watch"]},
    {
        "apiGroups": ["scheduling.volcano.sh"],
        "resources": ["queues", "podgroups"],
        "verbs": ["get", "list", "watch"],
    },
]

_NAS_MOUNT_OPTIONS = [
    "vers=3", "nolock", "proto=tcp", "rsize=1048576", "wsize=1048576",
    "hard", "timeo=600", "retrans=2", "noresvport",
]


def _ignore_conflict(fn, *args, **kwargs) -> bool:
    """Call a create-* API fn, swallowing 409 (already exists). Returns created?."""
    try:
        fn(*args, **kwargs)
        return True
    except ApiException as exc:
        if exc.status == 409:
            return False
        raise WujiError(humanize_api_exception(exc)) from exc


def check_admin() -> bool:
    """Return True if the caller can create namespaces AND clusterrolebindings.

    This is the proxy for "is an admin kubeconfig" used to gate ``register`` /
    ``set`` / ``set-queue``. Non-raising: any denial or API error → False.
    """
    auth = client.AuthorizationV1Api()
    for group, resource, verb in (
        ("", "namespaces", "create"),
        ("rbac.authorization.k8s.io", "clusterrolebindings", "create"),
    ):
        review = client.V1SelfSubjectAccessReview(
            spec=client.V1SelfSubjectAccessReviewSpec(
                resource_attributes=client.V1ResourceAttributes(
                    verb=verb, group=group, resource=resource
                )
            )
        )
        try:
            resp = auth.create_self_subject_access_review(review)
        except ApiException:
            return False
        if not (resp.status and resp.status.allowed):
            return False
    return True


def _require_admin() -> None:
    """Gate: raise unless the caller is an admin (see :func:`check_admin`).

    Fail fast with a clear message instead of leaving a half-created team when a
    non-admin runs an admin-only command.
    """
    if not check_admin():
        raise WujiError(
            "该命令需要管理员 kubeconfig(能创建 namespace / clusterrolebinding)。"
            "当前身份权限不足——请用集群管理员的 kubeconfig。"
        )


def whoami() -> Dict[str, Any]:
    """Report the caller's identity, admin status, namespace and default queue.

    Returns ``{namespace, is_admin, default_queue, team}`` — all best-effort
    (fields fall back to None/"?" if they can't be resolved).
    """
    load_clients()
    try:
        admin = check_admin()
    except Exception:  # noqa: BLE001
        admin = False
    try:
        ns = current_namespace(None)
    except Exception:  # noqa: BLE001 - admin kubeconfig may have no default ns
        ns = None
    team = None
    if ns:
        try:
            core, _, _ = load_clients()
            obj = core.read_namespace(name=ns)
            team = ((obj.metadata.labels or {}) if obj.metadata else {}).get(TEAM_LABEL)
        except Exception:  # noqa: BLE001
            team = None
    return {
        "namespace": ns,
        "is_admin": admin,
        "team": team,
        "default_queue": team or "default",
    }


def _nas_mkdir(core, team: str, nas_server: str, nas_base: str, timeout: int = 90) -> None:
    """Create the team's NAS sub-directory via a short-lived helper Pod.

    Mirrors add-team.sh step 2, but uses the pre-warmed ``ubuntu:24.04`` image
    (the old busybox ref was a Hangzhou VPC image edge nodes can't pull).
    Best-effort: NFS root must allow the mkdir; failures surface as WujiError.
    """
    pod_name = f"volcano-mkdir-{team}"
    try:
        core.delete_namespaced_pod(name=pod_name, namespace=team)
    except ApiException:
        pass
    body = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {"app.kubernetes.io/managed-by": "wuji"},
        },
        "spec": {
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "mkdir",
                    "image": "ubuntu:24.04",
                    "command": [
                        "sh", "-c",
                        f"mkdir -p /nas/{team} && chmod 0777 /nas/{team} && echo MKDIR-OK",
                    ],
                    "volumeMounts": [{"name": "nas", "mountPath": "/nas"}],
                }
            ],
            "volumes": [{"name": "nas", "nfs": {"server": nas_server, "path": nas_base}}],
        },
    }
    try:
        core.create_namespaced_pod(namespace=team, body=body)
    except ApiException as exc:
        raise WujiError(
            f"建 NAS 目录的 helper Pod 创建失败:{humanize_api_exception(exc)}"
        ) from exc
    deadline = time.time() + timeout
    phase = "Pending"
    while time.time() < deadline:
        try:
            pod = core.read_namespaced_pod(name=pod_name, namespace=team)
            phase = (pod.status.phase if pod.status else "Pending") or "Pending"
        except ApiException:
            phase = "Pending"
        if phase in ("Succeeded", "Failed"):
            break
        time.sleep(2)
    log = ""
    try:
        log = core.read_namespaced_pod_log(name=pod_name, namespace=team) or ""
    except ApiException:
        pass
    try:
        core.delete_namespaced_pod(name=pod_name, namespace=team)
    except ApiException:
        pass
    if phase != "Succeeded" or "MKDIR-OK" not in log:
        raise WujiError(
            f"在 NAS 上建团队目录失败(phase={phase})。请确认 NAS {nas_server}:{nas_base} "
            f"可写、路径存在。helper 输出:{log.strip()[:200] or '(空)'}"
        )


def register_team(
    team: str,
    *,
    apiserver: str = DEFAULT_APISERVER,
    nas_server: str = NAS_SERVER,
    nas_base: str = NAS_BASE,
    storage: str = "10Ti",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Admin-only: onboard a team end-to-end and return its ready-to-use kubeconfig.

    Creates (all idempotent, non-destructive — existing objects are left as-is):
    namespace, NAS ``<team>-nas-pv``/``<team>-nas`` (RWX, mounted at /workspace),
    ServiceAccount ``<team>``, Role ``<team>-volcano`` + RoleBinding, shared
    ClusterRole ``volcano-cluster-viewer`` + per-team ClusterRoleBinding, and a
    long-lived token Secret. Builds a kubeconfig (embedded CA + token, default
    namespace = team, server = ``apiserver``) and returns it as a YAML string.

    ACR is intentionally NOT touched: the platform credential-helper injects
    ``acr-credential-secret-aggregation`` into every namespace and binds it to the
    default ServiceAccount, so image pulls work without this command handling any
    registry credential.

    Args:
        team: team name = namespace (must be a valid DNS-1123 label).
        apiserver: API server URL baked into the generated kubeconfig.
        nas_server/nas_base: NFS coordinates; the team dir is ``<nas_base>/<team>``.
        storage: PV/PVC capacity string. dry_run: only return the plan, create nothing.

    Returns:
        ``{team, namespace, kubeconfig, apiserver, nas_pvc, created: [...], plan?}``.
        When ``dry_run`` is True, ``kubeconfig`` is empty and ``plan`` lists the
        objects that would be created.

    Raises:
        WujiError: on invalid name, insufficient privilege, or any API failure.
    """
    team = (team or "").strip()
    if not _TEAM_NAME_RE.match(team) or len(team) > 63:
        raise WujiError(
            f"团队名 {team!r} 非法:须是合法 k8s 名(小写字母/数字/-,以字母数字开头结尾,≤63 字符)。"
        )

    plan = [
        f"namespace/{team}",
        f"persistentvolume/{team}-nas-pv  (NFS {nas_server}:{nas_base}/{team}, {storage}, RWX)",
        f"persistentvolumeclaim/{team}/{team}-nas  -> /workspace",
        f"serviceaccount/{team}/{team}",
        f"role/{team}/{team}-volcano  (+ rolebinding)",
        f"clusterrole/{_CLUSTER_VIEWER_NAME} (shared) + clusterrolebinding/{team}-volcano-viewer",
        f"secret/{team}/{team}-token  (long-lived SA token)",
        f"kubeconfig  (server={apiserver}, ns={team})",
    ]
    if dry_run:
        return {
            "team": team, "namespace": team, "kubeconfig": "",
            "apiserver": apiserver, "nas_pvc": f"{team}-nas",
            "created": [], "plan": plan,
        }

    load_clients()  # ensure kube config is loaded before building extra clients
    _require_admin()
    core = client.CoreV1Api()
    rbac = client.RbacAuthorizationV1Api()
    created: List[str] = []

    # 1) namespace (must exist before namespaced objects / helper pod)
    if _ignore_conflict(core.create_namespace, {"metadata": {"name": team}}):
        created.append(f"namespace/{team}")

    # 2) NAS dir + PV + PVC
    _nas_mkdir(core, team, nas_server, nas_base)
    pv_body = {
        "apiVersion": "v1", "kind": "PersistentVolume",
        "metadata": {"name": f"{team}-nas-pv"},
        "spec": {
            "capacity": {"storage": storage},
            "accessModes": ["ReadWriteMany"],
            "persistentVolumeReclaimPolicy": "Retain",
            "mountOptions": list(_NAS_MOUNT_OPTIONS),
            "nfs": {"server": nas_server, "path": f"{nas_base}/{team}"},
        },
    }
    if _ignore_conflict(core.create_persistent_volume, pv_body):
        created.append(f"pv/{team}-nas-pv")
    pvc_body = {
        "apiVersion": "v1", "kind": "PersistentVolumeClaim",
        "metadata": {"name": f"{team}-nas", "namespace": team},
        "spec": {
            "accessModes": ["ReadWriteMany"],
            "resources": {"requests": {"storage": storage}},
            "volumeName": f"{team}-nas-pv",
        },
    }
    if _ignore_conflict(core.create_namespaced_persistent_volume_claim, team, pvc_body):
        created.append(f"pvc/{team}-nas")

    # 3) SA + Role + RoleBinding
    if _ignore_conflict(core.create_namespaced_service_account, team, {"metadata": {"name": team}}):
        created.append(f"sa/{team}")
    role_body = {
        "metadata": {"name": f"{team}-volcano", "namespace": team},
        "rules": _TEAM_ROLE_RULES,
    }
    if _ignore_conflict(rbac.create_namespaced_role, team, role_body):
        created.append(f"role/{team}-volcano")
    rb_body = {
        "metadata": {"name": f"{team}-volcano", "namespace": team},
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": f"{team}-volcano",
        },
        "subjects": [{"kind": "ServiceAccount", "name": team, "namespace": team}],
    }
    if _ignore_conflict(rbac.create_namespaced_role_binding, team, rb_body):
        created.append(f"rolebinding/{team}-volcano")

    # 4) shared read-only ClusterRole + per-team binding
    # reconcile(非仅 create):已存在则 patch rules——否则新增的权限(如 namespaces 只读,
    # 供默认队列路由读 ns 的 team 标签)永远不会应用到既有 ClusterRole,默认路由静默失效。
    cr_body = {"metadata": {"name": _CLUSTER_VIEWER_NAME}, "rules": _CLUSTER_VIEWER_RULES}
    try:
        rbac.create_cluster_role(cr_body)
    except ApiException as exc:
        if exc.status == 409:
            try:
                rbac.patch_cluster_role(
                    name=_CLUSTER_VIEWER_NAME, body={"rules": _CLUSTER_VIEWER_RULES}
                )
            except ApiException as exc2:
                raise WujiError(humanize_api_exception(exc2)) from exc2
        else:
            raise WujiError(humanize_api_exception(exc)) from exc
    crb_body = {
        "metadata": {"name": f"{team}-volcano-viewer"},
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole",
            "name": _CLUSTER_VIEWER_NAME,
        },
        "subjects": [{"kind": "ServiceAccount", "name": team, "namespace": team}],
    }
    if _ignore_conflict(rbac.create_cluster_role_binding, crb_body):
        created.append(f"clusterrolebinding/{team}-volcano-viewer")

    # 5) long-lived token Secret (k8s 1.24+ doesn't auto-create SA token secrets)
    sec_body = {
        "apiVersion": "v1", "kind": "Secret",
        "metadata": {
            "name": f"{team}-token", "namespace": team,
            "annotations": {"kubernetes.io/service-account.name": team},
        },
        "type": "kubernetes.io/service-account-token",
    }
    if _ignore_conflict(core.create_namespaced_secret, team, sec_body):
        created.append(f"secret/{team}-token")

    token = ca_b64 = ""
    for _ in range(20):  # controller injects token/ca into the secret asynchronously
        try:
            sec = core.read_namespaced_secret(name=f"{team}-token", namespace=team)
        except ApiException:
            time.sleep(1)
            continue
        data = sec.data or {}
        if data.get("token"):
            token = base64.b64decode(data["token"]).decode("utf-8")
            ca_b64 = data.get("ca.crt", "")  # already base64 (PEM) — kubeconfig wants b64
            break
        time.sleep(1)
    if not token:
        raise WujiError(
            f"团队资源已建,但 token 未注入 Secret {team}-token(稍后重跑本命令会复用已建资源并生成 kubeconfig)。"
        )

    # 6) assemble kubeconfig
    kubeconfig = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {"name": "wuji", "cluster": {"server": apiserver, "certificate-authority-data": ca_b64}}
        ],
        "contexts": [
            {"name": team, "context": {"cluster": "wuji", "namespace": team, "user": team}}
        ],
        "current-context": team,
        "users": [{"name": team, "user": {"token": token}}],
    }
    import yaml  # provided transitively by kubernetes

    return {
        "team": team,
        "namespace": team,
        "kubeconfig": yaml.safe_dump(kubeconfig, sort_keys=False),
        "apiserver": apiserver,
        "nas_pvc": f"{team}-nas",
        "created": created,
    }


# --------------------------------------------------------------------------- #
# two-layer resource limits: per-person ResourceQuota + per-team Volcano Queue
# --------------------------------------------------------------------------- #
# 模型:ns=一个人、queue=一个团队。个人卡上限 = 该 ns 的 ResourceQuota(admission 硬拦,
# 任何调度器都生效);团队总量 = Volcano Queue 的 deserved/capability(调度层,按队列记账)。
# `volcano set` 管前者,`volcano set-queue` 管后者;submit 默认把任务投到本人所属团队队列
# (读 ns 的 wuji.io/team 标签)。

_QUOTA_NAME = "gpu-quota"


def resolve_default_queue(team: Optional[str] = None) -> str:
    """Resolve the queue a submission should default to = the person's team.

    Reads the ``wuji.io/team`` label on the current namespace and uses it as the
    queue — **but only if a Queue by that name actually exists**; otherwise (or if
    the label is absent / the namespace can't be read) falls back to ``"default"``.
    Validating existence means labelling a namespace with a team that has no
    dedicated queue (e.g. under the single-``default``-queue model) never routes a
    submit to a non-existent queue (which Volcano would reject).
    """
    try:
        ns = current_namespace(team)
        core, _, custom = load_clients()
        obj = core.read_namespace(name=ns)
        labels = (obj.metadata.labels or {}) if obj.metadata else {}
        q = labels.get(TEAM_LABEL)
        if not q or q == "default":
            return "default"
        try:
            custom.get_cluster_custom_object(
                group=VOLCANO_QUEUE_GROUP, version=VOLCANO_QUEUE_VERSION,
                plural=VOLCANO_QUEUE_PLURAL, name=q,
            )
            return q  # queue exists → route there
        except ApiException:
            return "default"  # labelled queue doesn't exist → don't break submit
    except Exception:  # noqa: BLE001 - never block submit on queue resolution
        return "default"


def _quota_hard(gpu, cpu, memory, pods) -> Dict[str, str]:
    """Assemble a ResourceQuota ``hard`` map from the given per-person limits."""
    hard: Dict[str, str] = {}
    if gpu is not None:
        hard["requests.nvidia.com/gpu"] = str(gpu)
        hard["limits.nvidia.com/gpu"] = str(gpu)
    if cpu is not None:
        hard["requests.cpu"] = str(cpu)
        hard["limits.cpu"] = str(cpu)
    if memory is not None:
        hard["requests.memory"] = str(memory)
        hard["limits.memory"] = str(memory)
    if pods is not None:
        hard["pods"] = str(pods)
    return hard


# --------------------------------------------------------------------------- #
# home-node soft affinity (punchlist §8) — a Gatekeeper Assign per namespace,
# injecting k8s-native spec.affinity.nodeAffinity.preferredDuringScheduling...
# into every pod in that ns. Matches the naming/shape already used by the 12
# real users' pre-existing Assigns (home-node-<namespace>) discovered live on
# this cluster — this isn't a new mechanism, just the first time volcano-cli
# itself can create/read/remove them instead of them being hand-managed.
#
# Deliberately NOT Volcano's own nodegroup scheduler plugin: verified against
# the actual plugin source across v1.10.0 (running here), v1.11.0, v1.12.0,
# and current master — none of them make node-group affinity genuinely soft.
# v1.10.0 has no `strict` argument at all (enabling it with no affinity
# configured anywhere hard-rejects every node — caused a real cluster-wide
# scheduling outage, immediately reverted). Even on master, once a queue has
# ANY nodeGroupAffinity (required or preferred), its tasks are hard-filtered
# to matching nodes — preferred vs required only changes score weight, not
# whether other nodes are viable. A real k8s preferredDuringScheduling... via
# Gatekeeper is the only mechanism that's actually scoring-only.
# --------------------------------------------------------------------------- #
def _home_node_assign_name(namespace: str) -> str:
    """Deterministic name of the Assign implementing a namespace's home-node."""
    return f"home-node-{namespace}"


def _home_node_assign_spec(node: str) -> Dict[str, Any]:
    return {
        "applyTo": [{"groups": [""], "kinds": ["Pod"], "versions": ["v1"]}],
        "location": "spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution",
        "match": {
            "scope": "Namespaced",
            "kinds": [{"apiGroups": [""], "kinds": ["Pod"]}],
        },
        "parameters": {
            "assign": {
                "value": [
                    {
                        "weight": 100,
                        "preference": {
                            "matchExpressions": [
                                {"key": "kubernetes.io/hostname", "operator": "In", "values": [node]}
                            ]
                        },
                    }
                ]
            },
            # lets a user override with their own pod-spec affinity instead of
            # being clobbered by the default — matches the pre-existing Assigns.
            "pathTests": [
                {
                    "subPath": "spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution",
                    "condition": "MustNotExist",
                }
            ],
        },
    }


def get_home_node(namespace: str) -> Optional[str]:
    """Read a namespace's current home-node soft-affinity target, if any is set."""
    _, _, custom = load_clients()
    try:
        obj = custom.get_cluster_custom_object(
            group=GATEKEEPER_MUTATIONS_GROUP,
            version=GATEKEEPER_ASSIGN_VERSION,
            plural=GATEKEEPER_ASSIGN_PLURAL,
            name=_home_node_assign_name(namespace),
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise WujiError(humanize_api_exception(exc)) from exc
    try:
        pref = obj["spec"]["parameters"]["assign"]["value"][0]["preference"]
        values = pref["matchExpressions"][0]["values"]
        return values[0] if values else None
    except (KeyError, IndexError, TypeError):
        return None


def set_home_node(namespace: str, node: str) -> None:
    """Create or update ``namespace``'s home-node Assign to point at ``node``."""
    _, _, custom = load_clients()
    name = _home_node_assign_name(namespace)
    spec = _home_node_assign_spec(node)
    spec["match"]["namespaces"] = [namespace]
    body = {
        "apiVersion": f"{GATEKEEPER_MUTATIONS_GROUP}/{GATEKEEPER_ASSIGN_VERSION}",
        "kind": "Assign",
        "metadata": {"name": name},
        "spec": spec,
    }
    try:
        custom.create_cluster_custom_object(
            group=GATEKEEPER_MUTATIONS_GROUP,
            version=GATEKEEPER_ASSIGN_VERSION,
            plural=GATEKEEPER_ASSIGN_PLURAL,
            body=body,
        )
    except ApiException as exc:
        if exc.status != 409:
            raise WujiError(humanize_api_exception(exc)) from exc
        try:
            custom.patch_cluster_custom_object(
                group=GATEKEEPER_MUTATIONS_GROUP,
                version=GATEKEEPER_ASSIGN_VERSION,
                plural=GATEKEEPER_ASSIGN_PLURAL,
                name=name,
                body={"spec": spec},
            )
        except ApiException as exc2:
            raise WujiError(humanize_api_exception(exc2)) from exc2


def clear_home_node(namespace: str) -> bool:
    """Delete ``namespace``'s home-node Assign, if any. Returns whether one existed."""
    _, _, custom = load_clients()
    try:
        custom.delete_cluster_custom_object(
            group=GATEKEEPER_MUTATIONS_GROUP,
            version=GATEKEEPER_ASSIGN_VERSION,
            plural=GATEKEEPER_ASSIGN_PLURAL,
            name=_home_node_assign_name(namespace),
        )
        return True
    except ApiException as exc:
        if exc.status == 404:
            return False
        raise WujiError(humanize_api_exception(exc)) from exc


def get_quota(namespace: str) -> Optional[Dict[str, Any]]:
    """Read a person's ResourceQuota (name ``gpu-quota``). None if unset.

    Returns ``{namespace, hard: {...}, used: {...}}`` (hard/used are quantity
    strings) or None when the namespace has no ``gpu-quota``.
    """
    core, _, _ = load_clients()
    try:
        rq = core.read_namespaced_resource_quota(name=_QUOTA_NAME, namespace=namespace)
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise WujiError(humanize_api_exception(exc)) from exc
    # 上限读 spec.hard(声明即在,刚 create/patch 后立即可见);已用读 status.used
    # (由 quota 控制器 reconcile 后才有,读不到就当 0)。
    spec = rq.spec
    status = rq.status
    return {
        "namespace": namespace,
        "hard": dict((spec.hard or {}) if spec else {}),
        "used": dict((status.used or {}) if status else {}),
    }


def set_quota(
    namespace: str,
    *,
    gpu: Optional[int] = None,
    cpu: Optional[str] = None,
    memory: Optional[str] = None,
    pods: Optional[int] = None,
    team: Optional[str] = None,
    clear: bool = False,
    home_node: Optional[str] = None,
    no_home_node: bool = False,
) -> Dict[str, Any]:
    """Admin-only: set/patch a person's per-namespace resource limits.

    Upserts a ResourceQuota named ``gpu-quota`` in ``namespace`` with the given
    hard limits (only the dimensions you pass are touched — partial updates merge).
    ``team`` also writes the ``wuji.io/team`` namespace label (person→team). With
    ``clear`` the ResourceQuota is deleted (no limit). With no limit flags it just
    reports the current quota (and applies ``team``/``home_node`` if given).

    ``home_node``/``no_home_node`` set or remove this namespace's home-node soft
    affinity (a Gatekeeper Assign, see :func:`set_home_node`) — independent of
    the quota/team fields above, so ``volcano set <ns> --team X --home-node Y``
    does both in one call as the natural "onboard/edit this person" command.

    Raises:
        WujiError: if the caller isn't an admin, or on any API failure.
    """
    load_clients()
    _require_admin()
    core = client.CoreV1Api()

    label_set = None
    if team is not None:
        try:
            core.patch_namespace(
                name=namespace, body={"metadata": {"labels": {TEAM_LABEL: team}}}
            )
            label_set = team
        except ApiException as exc:
            raise WujiError(humanize_api_exception(exc)) from exc

    if home_node and no_home_node:
        raise WujiError("--home-node 和 --no-home-node 只能给一个。")
    if no_home_node:
        clear_home_node(namespace)
    elif home_node:
        set_home_node(namespace, home_node)
    home_node_now = get_home_node(namespace)

    if clear:
        try:
            core.delete_namespaced_resource_quota(name=_QUOTA_NAME, namespace=namespace)
        except ApiException as exc:
            if exc.status != 404:
                raise WujiError(humanize_api_exception(exc)) from exc
        return {
            "namespace": namespace, "cleared": True, "team": label_set,
            "quota": None, "home_node": home_node_now,
        }

    hard = _quota_hard(gpu, cpu, memory, pods)
    if not hard:
        # nothing to limit — just report current (and note any team/home-node change)
        return {
            "namespace": namespace, "team": label_set,
            "quota": get_quota(namespace), "home_node": home_node_now,
        }

    body = {
        "metadata": {"name": _QUOTA_NAME, "namespace": namespace},
        "spec": {"hard": hard},
    }
    try:
        core.create_namespaced_resource_quota(namespace=namespace, body=body)
    except ApiException as exc:
        if exc.status == 409:
            try:
                core.patch_namespaced_resource_quota(
                    name=_QUOTA_NAME, namespace=namespace, body={"spec": {"hard": hard}}
                )
            except ApiException as exc2:
                raise WujiError(humanize_api_exception(exc2)) from exc2
        else:
            raise WujiError(humanize_api_exception(exc)) from exc
    return {
        "namespace": namespace, "team": label_set,
        "quota": get_quota(namespace), "home_node": home_node_now,
    }


def get_queue_one(name: str) -> Optional[Dict[str, Any]]:
    """Read a single Volcano Queue's deserved/capability/reclaimable/state. None if absent."""
    _, _, custom = load_clients()
    try:
        obj = custom.get_cluster_custom_object(
            group=VOLCANO_QUEUE_GROUP,
            version=VOLCANO_QUEUE_VERSION,
            plural=VOLCANO_QUEUE_PLURAL,
            name=name,
        )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise WujiError(humanize_api_exception(exc)) from exc
    spec = obj.get("spec", {}) or {}
    st = obj.get("status", {}) or {}
    return {
        "name": name,
        "deserved": (spec.get("deserved") or {}).get("nvidia.com/gpu", "-"),
        "capability": (spec.get("capability") or {}).get("nvidia.com/gpu", "-"),
        "reclaimable": spec.get("reclaimable"),
        "state": st.get("state", "-"),
    }


def set_queue(
    name: str,
    *,
    deserved: Optional[int] = None,
    cap: Optional[int] = None,
    reclaimable: Optional[bool] = None,
    delete: bool = False,
) -> Dict[str, Any]:
    """Admin-only: create/update (or delete) a team's Volcano Queue.

    Upserts the cluster-scoped Queue ``name`` (= team), setting only the fields you
    pass (``deserved``/``cap`` are the ``nvidia.com/gpu`` quantities; ``reclaimable``
    toggles idle-borrow/reclaim). New queues default to ``reclaimable: true``. With
    ``delete`` the queue is removed (Volcano forbids deleting ``default``).

    Raises:
        WujiError: if the caller isn't an admin, or on any API failure.
    """
    load_clients()
    # 无任何改动参数 = 只查看,不写、也不需要 admin(避免拼错队列名把空队列建出来)。
    if not delete and deserved is None and cap is None and reclaimable is None:
        return {"name": name, "queue": get_queue_one(name)}
    _require_admin()
    _, _, custom = load_clients()

    if delete:
        try:
            custom.delete_cluster_custom_object(
                group=VOLCANO_QUEUE_GROUP,
                version=VOLCANO_QUEUE_VERSION,
                plural=VOLCANO_QUEUE_PLURAL,
                name=name,
            )
        except ApiException as exc:
            raise WujiError(humanize_api_exception(exc)) from exc
        return {"name": name, "deleted": True}

    try:
        existing = custom.get_cluster_custom_object(
            group=VOLCANO_QUEUE_GROUP,
            version=VOLCANO_QUEUE_VERSION,
            plural=VOLCANO_QUEUE_PLURAL,
            name=name,
        )
    except ApiException as exc:
        if exc.status == 404:
            existing = None
        else:
            raise WujiError(humanize_api_exception(exc)) from exc

    spec = dict(existing.get("spec", {})) if existing else {"reclaimable": True}
    if deserved is not None:
        spec["deserved"] = {**(spec.get("deserved") or {}), "nvidia.com/gpu": str(deserved)}
    if cap is not None:
        spec["capability"] = {**(spec.get("capability") or {}), "nvidia.com/gpu": str(cap)}
    if reclaimable is not None:
        spec["reclaimable"] = reclaimable

    body = {
        "apiVersion": f"{VOLCANO_QUEUE_GROUP}/{VOLCANO_QUEUE_VERSION}",
        "kind": "Queue",
        "metadata": {"name": name},
        "spec": spec,
    }
    try:
        if existing:
            body["metadata"]["resourceVersion"] = (
                existing.get("metadata", {}).get("resourceVersion")
            )
            custom.replace_cluster_custom_object(
                group=VOLCANO_QUEUE_GROUP,
                version=VOLCANO_QUEUE_VERSION,
                plural=VOLCANO_QUEUE_PLURAL,
                name=name,
                body=body,
            )
        else:
            custom.create_cluster_custom_object(
                group=VOLCANO_QUEUE_GROUP,
                version=VOLCANO_QUEUE_VERSION,
                plural=VOLCANO_QUEUE_PLURAL,
                body=body,
            )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc
    return {"name": name, "created": existing is None, "queue": get_queue_one(name)}


# --------------------------------------------------------------------------- #
# grant-admin — 【管理员】把某人设为平台管理员(能用 register/set/set-queue/...)
# --------------------------------------------------------------------------- #
# 曾经有一个收敛版 "volcano-admin" ClusterRole(只给 register/set/set-queue 所需的
# 最小规则,--cluster-admin 才给内置 cluster-admin 全权)。用户明确否决了这个两档
# 设计(punchlist §10):"我好像从来没明确说过要拆两种 admin,两种 admin 就应该是
# 一样的,admin 就应该有所有权限才对"——所以只保留一档,grant-admin 现在总是绑内置
# 的 cluster-admin,不再自建/维护一个单独的 ClusterRole。

# 审计记录(punchlist §4):grant-admin 曾经"悄悄发生"过(zhaoqingzeng 那次真实 cluster-admin
# 授权,直到主动排查才发现)。独立于 rl_cluster_console 的 kill-audit-log —— 那个是那个 bot
# 用它自己本地 kubeconfig 写的,权限模型不同,不能混用同一个 ConfigMap。
_ADMIN_GRANT_AUDIT_NS = "wuji-system"
_ADMIN_GRANT_AUDIT_CM = "admin-grant-audit-log"


def _current_operator_identity() -> str:
    """Best-effort: caller's own verified k8s identity (SelfSubjectReview username).

    Only used for the "谁做的" audit column — a failed lookup must not block an
    already-authorized admin action, so this degrades to "unknown" instead of
    raising (_require_admin already gated permission; this is just attribution).
    """
    try:
        auth = client.AuthenticationV1Api()
        resp = auth.create_self_subject_review(client.V1SelfSubjectReview())
        return resp.status.user_info.username
    except Exception:  # noqa: BLE001
        return "unknown"


def _log_admin_grant_audit(
    core, *, action: str, namespace: str, sa: Optional[str], role: Optional[str], reason: str
) -> bool:
    """Append one entry to wuji-system/admin-grant-audit-log (create-or-append).

    Best-effort: the RBAC change this records has already happened by the time
    this runs, so a failure here must not make grant_admin() look like it failed
    (that would be more confusing than a missing audit line) — returns False on
    failure instead of raising, so the caller can surface a warning without
    treating the whole operation as failed.
    """
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "operator": _current_operator_identity(),
        "action": action,
        "namespace": namespace,
        "sa": sa,
        "role": role,
        "reason": reason,
    }
    try:
        try:
            cm = core.read_namespaced_config_map(
                name=_ADMIN_GRANT_AUDIT_CM, namespace=_ADMIN_GRANT_AUDIT_NS
            )
            log = json.loads((cm.data or {}).get("log", "[]"))
            log.append(entry)
            core.patch_namespaced_config_map(
                name=_ADMIN_GRANT_AUDIT_CM,
                namespace=_ADMIN_GRANT_AUDIT_NS,
                body={"data": {"log": json.dumps(log, ensure_ascii=False)}},
            )
        except ApiException as exc:
            if exc.status != 404:
                raise
            body = {
                "metadata": {"name": _ADMIN_GRANT_AUDIT_CM, "namespace": _ADMIN_GRANT_AUDIT_NS},
                "data": {"log": json.dumps([entry], ensure_ascii=False)},
            }
            core.create_namespaced_config_map(namespace=_ADMIN_GRANT_AUDIT_NS, body=body)
        return True
    except Exception:  # noqa: BLE001
        return False


_ADMIN_ROLE = "cluster-admin"


def grant_admin(
    namespace: str,
    *,
    reason: str = "",
    sa: Optional[str] = None,
    revoke: bool = False,
) -> Dict[str, Any]:
    """Admin-only: make a person (their namespace's SA) a platform admin.

    Binds ``<namespace>``'s ServiceAccount to the built-in ``cluster-admin``
    ClusterRole via a ClusterRoleBinding ``<namespace>-admin``. Their existing
    kubeconfig gains admin power immediately (same token). With ``revoke`` it
    removes the binding. There is only one admin tier — see punchlist §10: a
    scoped-down "volcano-admin" role used to exist here, but the user explicitly
    rejected that two-tier design ("admin should just have all permissions").

    Every call (grant, revoke, or idempotent re-grant) appends an entry to
    ``wuji-system/admin-grant-audit-log`` (who/whom/when/why) — this used to be
    the one place admin power changed hands "silently"; see punchlist §4.

    Args:
        namespace: the person's namespace (its SA, named ``sa`` or ``namespace``,
            is the subject). reason: mandatory, human-readable justification —
            goes straight into the audit entry. sa: SA name override (defaults to
            ``namespace``). revoke: remove admin instead of granting.

    Returns:
        ``{namespace, sa, role, binding, audit_logged}`` on grant,
        ``{namespace, revoked: True, audit_logged}`` on revoke.

    Raises:
        WujiError: if ``reason`` is blank, the caller isn't an admin, or on any
            API failure.
    """
    namespace = (namespace or "").strip()
    if not _TEAM_NAME_RE.match(namespace) or len(namespace) > 63:
        raise WujiError(f"namespace {namespace!r} 非法(须是合法 k8s 名)。")
    sa = (sa or namespace).strip()
    if not _TEAM_NAME_RE.match(sa) or len(sa) > 63:
        raise WujiError(f"--sa {sa!r} 非法(须是合法 k8s 名)。")
    reason = (reason or "").strip()
    if not reason:
        raise WujiError(
            "--reason 不能为空:grant-admin 曾经悄悄发生过、没人知道,现在强制要求写明理由,"
            "会连同操作者、目标、时间一起记进 wuji-system/admin-grant-audit-log。"
        )
    core, _, _ = load_clients()
    _require_admin()
    rbac = client.RbacAuthorizationV1Api()
    crb_name = f"{namespace}-admin"

    if revoke:
        try:
            rbac.delete_cluster_role_binding(name=crb_name)
            existed = True
        except ApiException as exc:
            if exc.status != 404:
                raise WujiError(humanize_api_exception(exc)) from exc
            existed = False
        audit_logged = _log_admin_grant_audit(
            core, action="revoke" if existed else "revoke_noop",
            namespace=namespace, sa=sa, role=None, reason=reason,
        )
        return {"namespace": namespace, "revoked": True, "audit_logged": audit_logged}

    # 先读现有绑定:roleRef 不可变,不能靠 create 吞 409 静默"更新"。已绑同角色=幂等;
    # 绑着别的角色(比如旧版 volcano-admin 遗留的绑定)=报错让先 --revoke,不静默改写。
    try:
        existing = rbac.read_cluster_role_binding(name=crb_name)
    except ApiException as exc:
        if exc.status == 404:
            existing = None
        else:
            raise WujiError(humanize_api_exception(exc)) from exc
    if existing is not None:
        cur_role = existing.role_ref.name if existing.role_ref else None
        if cur_role == _ADMIN_ROLE:
            audit_logged = _log_admin_grant_audit(
                core, action="grant_unchanged", namespace=namespace, sa=sa,
                role=_ADMIN_ROLE, reason=reason,
            )
            return {"namespace": namespace, "sa": sa, "role": _ADMIN_ROLE,
                    "binding": crb_name, "unchanged": True, "audit_logged": audit_logged}
        raise WujiError(
            f"{namespace} 已绑定管理员角色 {cur_role};要改成 {_ADMIN_ROLE} 需先撤销再授:"
            f"`volcano grant-admin {namespace} --revoke` 然后重新 grant(roleRef 不可变)。"
        )

    crb_body = {
        "metadata": {"name": crb_name},
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": _ADMIN_ROLE,
        },
        "subjects": [{"kind": "ServiceAccount", "name": sa, "namespace": namespace}],
    }
    try:
        rbac.create_cluster_role_binding(crb_body)
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc
    audit_logged = _log_admin_grant_audit(
        core, action="grant", namespace=namespace, sa=sa, role=_ADMIN_ROLE, reason=reason,
    )
    return {"namespace": namespace, "sa": sa, "role": _ADMIN_ROLE, "binding": crb_name,
            "audit_logged": audit_logged}


def list_teams() -> List[Dict[str, Any]]:
    """List teams, aggregated by their ``wuji.io/team`` label — one row per team.

    ns=person, queue=team, so several people's namespaces can share one team;
    grouping by namespace (the old behavior) produced one row per *person* while
    calling the column "团队", which was misleading. Namespaces missing the label
    are never merged with each other (they don't actually share a team) — each
    becomes its own single-member group with ``team=""``.

    Per team returns ``{team, namespaces, gpu_quota, gpu_used, pods, home_nodes}``:
      - ``namespaces``: sorted member namespace names.
      - ``gpu_quota``: sum of each member's ``gpu-quota`` ResourceQuota GPU hard
        limit ("-" if no member has a readable numeric quota — quota read is
        cluster-wide, admin-only).
      - ``gpu_used``/``pods``: summed live GPU/pod count across members (from
        pods, which the shared cluster-viewer role can read, so ordinary users
        see this too).
      - ``home_nodes``: ``{namespace: node_or_None}``. Home-node (punchlist §8)
        is a per-namespace Gatekeeper Assign, not a per-team setting — a team's
        members can each be pinned to a different node, or none — so this is
        reported per member rather than collapsed to one value. Reading it needs
        cluster-scoped Assign read, which ordinary users generally don't have;
        degrades to ``None`` per-namespace on any error rather than failing the
        whole call.

    Reading namespaces + pods only needs the ``volcano-cluster-viewer`` role, so
    this works for any registered user; the quota and home-node columns degrade
    to "-"/``None`` when the caller can't read them cluster-wide.
    """
    core, _, _ = load_clients()
    try:
        nss = core.list_namespace(label_selector=TEAM_LABEL)
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc

    used: Dict[str, Dict[str, Any]] = {}
    try:
        for u in usage_by_namespace(None):
            used[u["namespace"]] = u
    except Exception:  # noqa: BLE001 - usage is best-effort
        pass

    quota: Dict[str, str] = {}
    try:
        rqs = core.list_resource_quota_for_all_namespaces()
        for r in rqs.items:
            if r.metadata.name == _QUOTA_NAME:
                hard = (r.spec.hard or {}) if r.spec else {}
                quota[r.metadata.namespace] = hard.get("requests.nvidia.com/gpu", "-")
    except ApiException:
        pass  # not permitted to list quotas cluster-wide → leave "-"

    groups: Dict[str, List[str]] = {}
    for ns in nss.items:
        name = ns.metadata.name
        labels = (ns.metadata.labels or {}) if ns.metadata else {}
        team = labels.get(TEAM_LABEL, "")
        key = team if team else f"\0{name}"  # untagged ns never merges with another
        groups.setdefault(key, []).append(name)

    out: List[Dict[str, Any]] = []
    for key, members in groups.items():
        members.sort()
        team = "" if key.startswith("\0") else key

        gpu_used = sum(int(used.get(m, {}).get("gpu", 0)) for m in members)
        pods = sum(int(used.get(m, {}).get("pods", 0)) for m in members)

        numeric_quotas = []
        for m in members:
            try:
                numeric_quotas.append(int(quota.get(m, "-")))
            except (TypeError, ValueError):
                pass
        gpu_quota = str(sum(numeric_quotas)) if numeric_quotas else "-"

        home_nodes: Dict[str, Optional[str]] = {}
        for m in members:
            try:
                home_nodes[m] = get_home_node(m)
            except Exception:  # noqa: BLE001 - best-effort, degrades to unknown
                home_nodes[m] = None

        out.append(
            {
                "team": team,
                "namespaces": members,
                "gpu_quota": gpu_quota,
                "gpu_used": gpu_used,
                "pods": pods,
                "home_nodes": home_nodes,
            }
        )
    out.sort(key=lambda t: (t["team"], t["namespaces"][0]))
    return out
