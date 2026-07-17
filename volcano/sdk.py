"""SDK functions — the programmatic surface of volcano.

Every function here is import-safe (``import volcano; volcano.submit(...)``) and is
also what ``volcano.cli`` calls under the hood. Each resolves the namespace,
loads clients, and turns Kubernetes ``ApiException``\\ s into :class:`WujiError`.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Dict, List, Optional, Union

from kubernetes.client.exceptions import ApiException

from .job import build_volcano_job
from .kube import (
    DEFAULT_ACR_REGISTRY,
    DEFAULT_ACR_REPO_PREFIX,
    IMAGE_INDEX_NAME,
    IMAGE_INDEX_NAMESPACE,
    JOB_NAME_LABEL,
    NODE_PUBLIC_IP_ANNOTATION,
    SAVE_REQUEST_LABEL,
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
    ssh_pubkey: Optional[str] = None,
    ssh_port: int = 22,
    ports: Optional[List[int]] = None,
    node: Optional[str] = None,
    kind: Optional[str] = None,
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
        The created object as returned by the API server. When ``expose_ssh`` is
        set, the dict carries an extra ``"_ssh_node_port"`` key with the
        allocated NodePort.

    Raises:
        WujiError: on any Kubernetes API failure (message is user-friendly).
    """
    ns = current_namespace(team)
    if expose_ssh:
        ssh = True  # exposing SSH is meaningless unless sshd is started
    # 保证:开了 SSH 但既没给密码也没给公钥时,自动生成强随机密码。否则 sshd 前置脚本
    # 不会启动(它要求 WUJI_SSH_PASSWORD 或 WUJI_SSH_PUBKEY 非空),而 NodePort 仍会建
    # → 一个对公网开放却无人监听的死端口。生成的密码回传给调用方显示一次。
    generated_password = None
    if (ssh or ssh_password or ssh_pubkey) and not ssh_password and not ssh_pubkey:
        generated_password = secrets.token_urlsafe(12)
        ssh_password = generated_password
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
        ssh_pubkey=ssh_pubkey,
        ssh_port=ssh_port,
        ports=ports,
        node=node,
        kind=kind,
    )
    _, _, custom = load_clients()
    try:
        created = custom.create_namespaced_custom_object(
            group=VOLCANO_JOB_GROUP,
            version=VOLCANO_JOB_VERSION,
            namespace=ns,
            plural=VOLCANO_JOB_PLURAL,
            body=manifest,
        )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc

    if isinstance(created, dict) and generated_password:
        created["_ssh_generated_password"] = generated_password
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
        all_namespaces: if True, list across every namespace (admin view;
            needs cluster-level list on volcano jobs) instead of one team ns.

    Returns:
        A list of ``{namespace, name, phase, queue, gpus, kind, age}`` summary dicts.
    """
    _, _, custom = load_clients()
    try:
        if all_namespaces:
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


def top_nodes() -> List[Dict[str, Any]]:
    """Per GPU-node resource usage: GPU/CPU/memory allocated vs capacity.

    "allocated" = sum of pod GPU limits and CPU/memory requests for
    Running/Pending pods on that node (what the scheduler has reserved).

    Returns:
        A list (sorted by node name) of dicts with keys ``node``,
        ``gpu_used/gpu_cap``, ``cpu_used_m/cpu_cap_m``, ``mem_used_gi/mem_cap_gi``.
        Only nodes that expose ``nvidia.com/gpu`` are included.
    """
    core, _, _ = load_clients()
    # kubernetes 客户端把上百个 pod 反序列化成 typed 对象极慢(~15s)。
    # 直接取原始 JSON(_preload_content=False)自己解析,并在服务端过滤掉已完成 pod。
    try:
        nresp = core.list_node(_preload_content=False)
        presp = core.list_pod_for_all_namespaces(
            field_selector="status.phase!=Succeeded,status.phase!=Failed",
            _preload_content=False,
        )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc
    nodes = json.loads(nresp.data).get("items", [])
    pods = json.loads(presp.data).get("items", [])

    used: Dict[str, Dict[str, float]] = {}
    for p in pods:
        spec = p.get("spec", {})
        node = spec.get("nodeName")
        phase = p.get("status", {}).get("phase")
        if not node or phase not in ("Running", "Pending"):
            continue
        acc = used.setdefault(node, {"gpu": 0, "cpu_m": 0.0, "mem_gi": 0.0})
        for c in spec.get("containers", []):
            res = c.get("resources") or {}
            req = res.get("requests") or {}
            lim = res.get("limits") or {}
            acc["gpu"] += _gpu(lim.get("nvidia.com/gpu"))
            acc["cpu_m"] += _cpu_to_m(req.get("cpu"))
            acc["mem_gi"] += _mem_to_gi(req.get("memory"))

    out: List[Dict[str, Any]] = []
    for nd in nodes:
        alloc = nd.get("status", {}).get("allocatable") or {}
        gpu_cap = _gpu(alloc.get("nvidia.com/gpu"))
        if gpu_cap == 0:
            continue  # only GPU nodes
        name = nd.get("metadata", {}).get("name", "?")
        u = used.get(name, {"gpu": 0, "cpu_m": 0.0, "mem_gi": 0.0})
        out.append(
            {
                "node": name,
                "gpu_used": int(u["gpu"]), "gpu_cap": gpu_cap,
                "cpu_used_m": u["cpu_m"], "cpu_cap_m": _cpu_to_m(alloc.get("cpu")),
                "mem_used_gi": u["mem_gi"], "mem_cap_gi": _mem_to_gi(alloc.get("memory")),
            }
        )
    out.sort(key=lambda x: x["node"])
    return out


def usage_by_namespace(team: Optional[str] = None) -> List[Dict[str, Any]]:
    """按 namespace(=人/团队)聚合 Running/Pending pod 的 GPU/CPU/内存用量。

    Args:
        team: 只看某个 namespace;为 None 时返回所有"在占 GPU"的 namespace。

    Returns:
        ``[{namespace, gpu, cpu_m, mem_gi, pods}]``,按 GPU 降序。
    """
    core, _, _ = load_clients()
    try:
        presp = core.list_pod_for_all_namespaces(
            field_selector="status.phase!=Succeeded,status.phase!=Failed",
            _preload_content=False,
        )
    except ApiException as exc:
        raise WujiError(humanize_api_exception(exc)) from exc
    pods = json.loads(presp.data).get("items", [])

    agg: Dict[str, Dict[str, float]] = {}
    for p in pods:
        ns = p.get("metadata", {}).get("namespace", "")
        if p.get("status", {}).get("phase") not in ("Running", "Pending"):
            continue
        a = agg.setdefault(ns, {"gpu": 0, "cpu_m": 0.0, "mem_gi": 0.0, "pods": 0})
        a["pods"] += 1
        for c in p.get("spec", {}).get("containers", []):
            res = c.get("resources") or {}
            req = res.get("requests") or {}
            lim = res.get("limits") or {}
            a["gpu"] += _gpu(lim.get("nvidia.com/gpu"))
            a["cpu_m"] += _cpu_to_m(req.get("cpu"))
            a["mem_gi"] += _mem_to_gi(req.get("memory"))

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
