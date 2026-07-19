"""Kubernetes client plumbing for volcano.

Handles the dual auth mode required by the platform:

* inside the cluster (a notebook / pod using its ServiceAccount) we use
  :func:`kubernetes.config.load_incluster_config`;
* on a user's laptop we fall back to their local kubeconfig via
  :func:`kubernetes.config.load_kube_config`.

It also centralises the Volcano CRD coordinates so the rest of the package
never has to remember group/version/plural strings.
"""

from __future__ import annotations

from typing import Optional, Tuple

from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException

# --- Volcano custom-resource coordinates -----------------------------------
# batch.volcano.sh/v1alpha1 Job  (the training job)
VOLCANO_JOB_GROUP = "batch.volcano.sh"
VOLCANO_JOB_VERSION = "v1alpha1"
VOLCANO_JOB_PLURAL = "jobs"

# scheduling.volcano.sh/v1beta1 Queue  (cluster-scoped)
VOLCANO_QUEUE_GROUP = "scheduling.volcano.sh"
VOLCANO_QUEUE_VERSION = "v1beta1"
VOLCANO_QUEUE_PLURAL = "queues"

# Pods created by a Volcano Job carry this label = the job name.
JOB_NAME_LABEL = "volcano.sh/job-name"

# Namespace label recording which team (= Volcano queue) a person's ns belongs to.
# `volcano set <ns> --team <t>` writes it; submit reads it to default --queue to the
# person's team queue (so ns=person, queue=team). Falls back to "default" if absent.
TEAM_LABEL = "wuji.io/team"

# Admin annotates each edge node with its elastic public IP so wuji can print a
# reachable ``ssh root@<ip> -p <nodeport>`` line for --expose-ssh jobs.
NODE_PUBLIC_IP_ANNOTATION = "wuji.io/public-ip"

# mutations.gatekeeper.sh/v1 Assign (cluster-scoped) — used for per-namespace
# home-node soft affinity (`volcano set <ns> --home-node <node>`). Volcano's own
# nodegroup scheduler plugin was evaluated for this and rejected: on this
# cluster's v1.10.0 it has no soft/optional mode at all, and even upstream
# master still hard-filters once a queue has any nodeGroupAffinity configured
# (see punchlist §8) — a real k8s-native `preferredDuringSchedulingIgnoredDuringExecution`
# injected via Gatekeeper is the only mechanism that's genuinely soft (scoring-only,
# never excludes other nodes).
GATEKEEPER_MUTATIONS_GROUP = "mutations.gatekeeper.sh"
GATEKEEPER_ASSIGN_VERSION = "v1"
GATEKEEPER_ASSIGN_PLURAL = "assign"

# --- volcano save (platform-delegated image commit) ---------------------------
# A `volcano save` request is just a ConfigMap in the user's namespace carrying
# this label; the privileged `wuji-saver` DaemonSet watches for it, commits the
# user's live container on the node, and pushes to ACR. The user never touches
# containerd or ACR write credentials — the saver holds both.
SAVE_REQUEST_LABEL = "wuji.io/save-request"

# Live ACR (Huhehaote, same region as the cluster). Saved dev images land here.
DEFAULT_ACR_REGISTRY = "wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com"
DEFAULT_ACR_REPO_PREFIX = "wuji-rl"

# Platform image index: a shared ConfigMap listing images users can pull. The ACR
# pull token can't list the catalog (pull-only scope), so instead wuji-saver
# appends each saved image here and admins seed base images. ``volcano images`` reads
# it — no ACR credential needed on the user side. All authenticated users are
# granted read on just this one ConfigMap (see saver/image-index.yaml).
IMAGE_INDEX_NAMESPACE = "wuji-public"
IMAGE_INDEX_NAME = "wuji-images"

# --- team onboarding (volcano register, admin-only) --------------------------
# Public apiserver endpoint baked into generated team kubeconfigs (edge nodes
# have no VPC route; teams reach the cluster via this elastic public endpoint).
DEFAULT_APISERVER = "https://39.104.72.117:6443"
# Bare-IP NFS NAS (no CSI on ENS). Each team gets a static PV at NAS_BASE/<team>,
# mounted at /workspace. These are infra coordinates, not secrets.
NAS_SERVER = "100.64.128.15"
NAS_BASE = "/1704065796538912/wuji-nas-0/wuji_ws_2/users"


class WujiError(RuntimeError):
    """User-facing error: message is safe to print without a traceback."""


def load_clients() -> Tuple[client.CoreV1Api, client.BatchV1Api, client.CustomObjectsApi]:
    """Load kube config (in-cluster first, then kubeconfig) and build API clients.

    Returns:
        A tuple ``(core_v1, batch_v1, custom)`` of ready-to-use API clients.

    Raises:
        WujiError: if neither in-cluster config nor a local kubeconfig can be
            loaded (e.g. running off-cluster with no ``~/.kube/config``).
    """
    try:
        config.load_incluster_config()
    except ConfigException:
        try:
            config.load_kube_config()
        except ConfigException as exc:  # no in-cluster SA and no kubeconfig
            raise WujiError(
                "找不到 Kubernetes 配置:集群内需挂载 ServiceAccount,"
                "集群外需要 ~/.kube/config(或设置 $KUBECONFIG)。\n"
                f"原始错误: {exc}"
            ) from exc

    return client.CoreV1Api(), client.BatchV1Api(), client.CustomObjectsApi()


def current_namespace(team: Optional[str] = None) -> str:
    """Resolve the namespace to operate in.

    Args:
        team: explicit team/namespace override (wins if provided).

    Resolution order:
        1. ``team`` argument (from ``--team``),
        2. the namespace of the active kubeconfig context,
        3. the in-cluster ServiceAccount namespace file.

    If none resolve, raises :class:`WujiError` (does NOT fall back to
    ``"default"`` — default has no ``<team>-nas`` volume and would FailedMount).

    Returns:
        The resolved namespace name.

    Raises:
        WujiError: when no team/context/in-cluster namespace can be determined.
    """
    if team:
        return team

    # 2) kubeconfig context namespace
    try:
        _, active = config.list_kube_config_contexts()
        ns = (active or {}).get("context", {}).get("namespace")
        if ns:
            return ns
    except (ConfigException, FileNotFoundError, IndexError, TypeError):
        pass

    # 3) in-cluster SA namespace
    try:
        with open(
            "/var/run/secrets/kubernetes.io/serviceaccount/namespace",
            encoding="utf-8",
        ) as fh:
            ns = fh.read().strip()
            if ns:
                return ns
    except OSError:
        pass

    # 4) 无法确定 namespace:不静默落到 "default"。default 没有 <team>-nas 数据卷,
    #    任务会 FailedMount 卡住且报错难懂,所以这里明确报错、引导用户指定团队。
    raise WujiError(
        "无法确定团队 namespace:请用 --team <团队> 指定,或在 kubeconfig 的 "
        "context 里设置 namespace。不要用没设默认 namespace 的 admin kubeconfig"
        "——否则任务会落到 default,而 default 没有 <团队>-nas 数据卷,Pod 会 "
        "FailedMount 卡住。"
    )


def humanize_api_exception(exc: Exception) -> str:
    """Turn a kubernetes ``ApiException`` (or any error) into a friendly line.

    Args:
        exc: the caught exception.

    Returns:
        A short human-readable message (no stack trace).
    """
    from kubernetes.client.exceptions import ApiException

    if isinstance(exc, ApiException):
        reason = exc.reason or "请求失败"
        detail = ""
        body = getattr(exc, "body", None)
        if body:
            try:
                import json

                detail = json.loads(body).get("message", "")
            except (ValueError, TypeError):
                detail = ""
        if exc.status == 401:
            return "认证失败(401):凭据无效或过期,请检查 kubeconfig / ServiceAccount。"
        if exc.status == 403:
            return f"权限不足(403):{detail or reason}。请确认账号在该 namespace 的 RBAC。"
        if exc.status == 404:
            return f"资源不存在(404):{detail or reason}。"
        if exc.status == 409:
            return f"资源冲突(409):{detail or reason}(可能同名任务已存在)。"
        return f"Kubernetes API 错误({exc.status}):{detail or reason}"

    return str(exc)
