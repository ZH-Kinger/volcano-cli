"""volcano command line interface (Typer).

Commands: ``submit``, ``list``/``ls``, ``status``, ``logs``, ``kill``,
``queue``, ``data ls``. Every command loads clients lazily so that
``volcano --help`` works with no kubeconfig and no network.
"""

from __future__ import annotations

import sys
import time
from typing import List, Optional

import typer

from .kube import WujiError, current_namespace, humanize_api_exception, load_clients
from . import sdk

app = typer.Typer(
    add_completion=False,
    help="volcano — 把训练任务提交成 Volcano Job(ACK Edge GPU 集群)。",
    no_args_is_help=True,
)
data_app = typer.Typer(help="查看 NAS / 共享数据集里的数据。", no_args_is_help=True)
app.add_typer(data_app, name="data")


# --------------------------------------------------------------------------- #
# output helpers (rich is optional)
# --------------------------------------------------------------------------- #
def _echo(msg: str = "") -> None:
    typer.echo(msg)


def _err(msg: str) -> None:
    typer.echo(f"错误: {msg}", err=True)


def _print_table(title: str, columns: List[str], rows: List[List[str]]) -> None:
    """Print a table with rich if available, else aligned plain text."""
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=title)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(c) for c in row])
        Console().print(table)
        return
    except ImportError:
        pass

    if title:
        _echo(title)
    if not rows:
        _echo("(空)")
        return
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    _echo(fmt.format(*columns))
    for row in rows:
        _echo(fmt.format(*[str(c) for c in row]))


def _die(exc: Exception) -> "typer.Exit":
    """Print a friendly error and exit non-zero (no traceback)."""
    _err(exc.args[0] if isinstance(exc, WujiError) else humanize_api_exception(exc))
    raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# login — 把用户的 kubeconfig 写到默认位置,之后免设 KUBECONFIG
# --------------------------------------------------------------------------- #
@app.command()
def login(
    file: Optional[str] = typer.Option(
        None, "--file", "-f",
        help="kubeconfig 文件路径;不给则从粘贴内容读(结束:Unix Ctrl-D / Windows Ctrl-Z 回车)。",
    ),
    dest: Optional[str] = typer.Option(None, "--dest", help="写入路径,默认 ~/.kube/config。"),
    force: bool = typer.Option(False, "--force", help="已存在时不备份直接覆盖。"),
) -> None:
    """把你的 kubeconfig 写到默认位置(~/.kube/config),之后 volcano/kubectl 免设 KUBECONFIG 直接用。

    用法:`volcano login -f team.kubeconfig`  或  `volcano login`(然后粘贴内容)。
    """
    import os
    import pathlib
    import shutil
    from time import localtime, strftime

    import yaml  # 由 kubernetes 依赖带入

    if file:
        try:
            content = pathlib.Path(file).read_text(encoding="utf-8")
        except OSError as exc:
            _err(f"读不到文件 {file}: {exc}")
            raise typer.Exit(code=1)
    else:
        _echo("粘贴你的 kubeconfig,结束按 Ctrl-D(Windows: Ctrl-Z 然后回车):")
        content = sys.stdin.read()

    try:
        doc = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        _err(f"内容不是合法 YAML: {exc}")
        raise typer.Exit(code=1)
    if not isinstance(doc, dict) or not all(
        k in doc for k in ("clusters", "contexts", "users")
    ):
        _err("这不像有效的 kubeconfig(缺 clusters/contexts/users)。")
        raise typer.Exit(code=1)

    target = pathlib.Path(dest) if dest else pathlib.Path.home() / ".kube" / "config"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not force:
            bak = target.with_name(
                target.name + ".bak." + strftime("%Y%m%d-%H%M%S", localtime())
            )
            shutil.copy2(target, bak)
            _echo(f"已备份原 kubeconfig -> {bak}")
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        _err(f"写入 {target} 失败: {exc}")
        raise typer.Exit(code=1)
    try:
        os.chmod(target, 0o600)  # POSIX 生效;Windows 上是 no-op
    except OSError:
        pass

    ctx_name = doc.get("current-context", "")
    ns = ""
    for c in doc.get("contexts", []) or []:
        if c.get("name") == ctx_name:
            ns = (c.get("context", {}) or {}).get("namespace", "")
            break
    _echo(f"✅ 已写入 {target}")
    _echo(f"   context={ctx_name or '(未设)'}  namespace={ns or '(未设)'}")
    _echo("现在直接 `volcano ls` / `volcano top` 即可,不用再设 KUBECONFIG。")
    if not ns:
        _echo("提示:该 kubeconfig 没设默认 namespace;非 admin 记得用 `-t <团队>`。")


# --------------------------------------------------------------------------- #
# whoami — 我是谁 / 是不是管理员 / 默认投哪个队列
# --------------------------------------------------------------------------- #
@app.command()
def whoami() -> None:
    """看当前 kubeconfig 的身份:namespace、是否管理员(能否用 register/set)、默认队列。"""
    try:
        info = sdk.whoami()
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    ns = info.get("namespace") or "(无默认 namespace)"
    _echo(f"namespace : {ns}")
    _echo(f"默认队列  : {info.get('default_queue')}"
          + (f"  (团队标签 wuji.io/team={info['team']})" if info.get("team") else "  (无团队标签→落 default)"))
    if info.get("is_admin"):
        _echo("身份      : ✅ 管理员(可用 volcano register / set / set-queue)")
    else:
        _echo("身份      : 普通用户(不能用 register / set / set-queue;只能管自己 ns 的任务)")


# --------------------------------------------------------------------------- #
# register — 【管理员】一键开通团队 + 生成 kubeconfig
# --------------------------------------------------------------------------- #
@app.command()
def register(
    team: str = typer.Argument(
        ..., help="团队名 = namespace(合法 k8s 名:小写字母/数字/-,以字母数字开头结尾)。"
    ),
    out: Optional[str] = typer.Option(
        None, "--out", "-o", help="kubeconfig 写入路径,默认 ./<团队>.kubeconfig。"
    ),
    apiserver: Optional[str] = typer.Option(
        None, "--apiserver", help="写进生成 kubeconfig 的 apiserver 地址(默认平台公网端点)。"
    ),
    storage: str = typer.Option("10Ti", "--storage", help="团队 NAS PV/PVC 容量。"),
    print_only: bool = typer.Option(
        False, "--print", help="把 kubeconfig 打到屏幕而非写文件。"
    ),
    force: bool = typer.Option(False, "--force", help="输出文件已存在时覆盖(默认拒绝)。"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印将创建的资源,不真建。"),
) -> None:
    """【管理员】一键开通一个团队并生成它的 kubeconfig。

    需要**管理员 kubeconfig**(能建 namespace / clusterrolebinding);普通团队身份会被拒。
    幂等:重复跑不会破坏已存在的资源,只补齐缺的并重新生成 kubeconfig。

    做的事(= 老 add-team.sh + grant-volcano-kubeconfig.sh):建 ns + NAS `<团队>-nas`
    PV/PVC(挂 /workspace)+ SA/Role/RoleBinding + 只读集群视图 + 长期 token → 生成
    `<团队>.kubeconfig`。**不碰 ACR**——平台已自动给每个 ns 注入拉取凭据。

    用法:`volcano register <团队>`,把生成的 kubeconfig 安全发给团队,对方 `volcano login -f` 即用。
    """
    kwargs: dict = {"storage": storage, "dry_run": dry_run}
    if apiserver:
        kwargs["apiserver"] = apiserver
    try:
        result = sdk.register_team(team, **kwargs)
    except Exception as exc:  # noqa: BLE001 - friendly exit
        _die(exc)

    if dry_run:
        _echo(f"[dry-run] 将为团队 {team} 创建(去掉 --dry-run 即真正执行):")
        for p in result.get("plan", []):
            _echo(f"  - {p}")
        return

    kc = result["kubeconfig"]
    if print_only:
        _echo(kc)
    else:
        import os
        import pathlib

        target = pathlib.Path(out) if out else pathlib.Path(f"./{team}.kubeconfig")
        if target.exists() and not force:
            _err(
                f"{target} 已存在;用 --force 覆盖、--out 换路径,或 --print 打到屏幕。"
            )
            raise typer.Exit(code=1)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(kc, encoding="utf-8")
        except OSError as exc:
            _err(f"写入 {target} 失败:{exc}")
            raise typer.Exit(code=1)
        try:
            os.chmod(target, 0o600)  # POSIX 生效;Windows 上是 no-op(靠 NTFS ACL)
        except OSError:
            pass
        _echo(f"✅ 团队 {team} 已开通。kubeconfig -> {target}")

    if result.get("created"):
        _echo("  新建资源: " + ", ".join(result["created"]))
    else:
        _echo("  (资源均已存在,复用;仅重新生成 kubeconfig)")
    _echo(f"  NAS: {result['nas_pvc']} → /workspace    apiserver: {result['apiserver']}")
    _echo(
        "分发:把该 kubeconfig 安全发给团队(含长期 token,勿群发/入 git);"
        "对方 `volcano login -f <文件>` 即可用。"
    )


# --------------------------------------------------------------------------- #
# set / set-queue — 【管理员】改个人(ns)资源上限 与 团队(queue)额度
# --------------------------------------------------------------------------- #
@app.command(name="set")
def set_cmd(
    namespace: str = typer.Argument(..., help="要设限的个人 namespace。"),
    gpu: Optional[int] = typer.Option(None, "--gpu", "-g", help="GPU 上限(张)。"),
    cpu: Optional[str] = typer.Option(None, "--cpu", "-c", help="CPU 上限(如 64)。"),
    memory: Optional[str] = typer.Option(None, "--memory", "-m", help="内存上限(如 256Gi)。"),
    pods: Optional[int] = typer.Option(None, "--pods", help="Pod 数上限。"),
    team: Optional[str] = typer.Option(
        None, "--team", help="把此人归属到某团队(写 ns 标签 wuji.io/team,决定默认队列)。"
    ),
    clear: bool = typer.Option(False, "--clear", help="删除该人的 ResourceQuota(取消限制)。"),
) -> None:
    """【管理员】设/改一个人(namespace)的资源上限(ResourceQuota),也可归属团队。

    不带任何限额参数 = 只查看当前上限与已用。个人上限是 admission 层硬限,任何调度器都生效;
    团队总量用 `volcano set-queue`。需要管理员 kubeconfig。
    """
    try:
        result = sdk.set_quota(
            namespace, gpu=gpu, cpu=cpu, memory=memory, pods=pods, team=team, clear=clear,
        )
    except Exception as exc:  # noqa: BLE001
        _die(exc)

    if result.get("team"):
        _echo(f"已归属团队: {namespace} → wuji.io/team={result['team']}(默认队列将投它)")
    if result.get("cleared"):
        _echo(f"✅ 已取消 {namespace} 的资源限制(删除 ResourceQuota)。")
        return
    q = result.get("quota")
    if not q or not q.get("hard"):
        _echo(f"{namespace} 当前无资源限制(没有 ResourceQuota)。")
        _echo("设限示例: volcano set " + namespace + " --gpu 8")
        return
    hard, used = q["hard"], q.get("used", {})
    rows = [[k, used.get(k, "0"), v] for k, v in sorted(hard.items())]
    _print_table(f"{namespace} 资源上限(ResourceQuota gpu-quota)", ["资源", "已用", "上限"], rows)


@app.command(name="set-queue")
def set_queue_cmd(
    team: str = typer.Argument(..., help="团队队列名(= 团队)。"),
    deserved: Optional[int] = typer.Option(
        None, "--deserved", "-d", help="保底/公平份额 GPU 数。"
    ),
    cap: Optional[int] = typer.Option(
        None, "--cap", "-c", help="突发上限 GPU 数(capability)。"
    ),
    reclaimable: Optional[bool] = typer.Option(
        None, "--reclaimable/--no-reclaimable", help="是否允许闲借忙收(默认建队时为开)。"
    ),
    delete: bool = typer.Option(False, "--delete", help="删除该队列(default 删不掉)。"),
) -> None:
    """【管理员】创建/修改(或删除)一个团队的 Volcano 队列额度。

    不带任何参数 = 只查看该队列现状。团队总量(队列下所有人加起来)受此约束;个人上限用
    `volcano set`。改完 vc-scheduler 自动重载,不影响在跑任务。需要管理员 kubeconfig。
    """
    try:
        result = sdk.set_queue(
            team, deserved=deserved, cap=cap, reclaimable=reclaimable, delete=delete,
        )
    except Exception as exc:  # noqa: BLE001
        _die(exc)

    if result.get("deleted"):
        _echo(f"✅ 已删除队列: {team}")
        return
    q = result.get("queue")
    if "created" not in result:
        # 查看模式(没给任何改动参数):只报现状,不谎报"已更新"
        if q:
            _echo(f"队列 {team}: deserved={q['deserved']} / cap={q['capability']} "
                  f"/ reclaimable={q['reclaimable']} / state={q['state']}")
        else:
            _echo(f"队列 {team} 不存在(用 --deserved/--cap 创建)。")
        return
    verb = "已创建" if result.get("created") else "已更新"
    if q:
        _echo(f"✅ {verb}队列 {team}: deserved={q['deserved']} / cap={q['capability']} "
              f"/ reclaimable={q['reclaimable']} / state={q['state']}")
    else:
        _echo(f"✅ {verb}队列 {team}。")


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
def _submit_impl(
    *,
    command: Optional[List[str]],
    cmd: Optional[str],
    name: str,
    team: Optional[str],
    queue: Optional[str],
    image: str,
    gpus: int,
    nodes: int,
    data: Optional[str],
    shared_data: Optional[str],
    cpu: str,
    memory: str,
    ssh: bool,
    ssh_password: Optional[str],
    ssh_pubkey: Optional[str],
    ssh_port: int,
    expose_ssh: bool,
    ssh_node_port: Optional[int],
    port: Optional[List[int]],
    node: Optional[str],
    mount_path: str,
    shell: str,
    dry_run: bool,
    kind: str,
    mounts: Optional[List[str]] = None,
) -> None:
    """``volcano train`` / ``volcano submit`` / ``volcano dev`` 的共用实现。"""
    run_cmd = list(command) if command else None
    if run_cmd and cmd:
        _err("--cmd 与 -- CMD 只能给一个。")
        raise typer.Exit(code=1)
    final_cmd = run_cmd if run_cmd else cmd
    if not final_cmd:
        _err("必须提供训练命令:用 --cmd '...' 或在末尾加 -- python train.py")
        raise typer.Exit(code=1)
    if mount_path == "/" or not mount_path.startswith("/"):
        _err("--mount-path 必须是绝对路径,且不能是 /(会覆盖容器根)。")
        raise typer.Exit(code=1)
    # 未显式 --queue 时,默认投到本人所属团队的队列(读 ns 的 wuji.io/team 标签);
    # 没标签/读不到则回退 default。让 ns=人、queue=团队 的模型对用户透明。
    if queue is None:
        queue = sdk.resolve_default_queue(team)
    if ssh_node_port is not None and not (30000 <= ssh_node_port <= 32767):
        _err("--ssh-nodeport 必须在 30000-32767 之间(Kubernetes NodePort 段)。")
        raise typer.Exit(code=1)
    if ssh_pubkey:
        import os

        if os.path.isfile(ssh_pubkey):  # 允许直接给公钥文件路径
            try:
                with open(ssh_pubkey, encoding="utf-8") as fh:
                    ssh_pubkey = fh.read().strip()
            except OSError as exc:
                _err(f"读取公钥文件失败: {exc}")
                raise typer.Exit(code=1)

    if dry_run:
        import yaml  # provided transitively by kubernetes

        from .job import build_volcano_job

        ns = current_namespace(team)
        try:
            manifest = build_volcano_job(
                name=name, team=ns, queue=queue, image=image, gpus=gpus,
                nodes=nodes, command=final_cmd, data=data, shared_data=shared_data,
                cpu=cpu, memory=memory, ssh=ssh, ssh_password=ssh_password,
                ssh_pubkey=ssh_pubkey, ssh_port=ssh_port, ports=port, node=node,
                workspace_path=mount_path, shell=shell, kind=kind, mounts=mounts,
            )
        except ValueError as exc:
            _die(WujiError(str(exc)))
        _echo(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
        return

    try:
        result = sdk.submit(
            name=name, team=team, queue=queue, image=image, command=final_cmd,
            gpus=gpus, nodes=nodes, data=data, shared_data=shared_data,
            cpu=cpu, memory=memory, ssh=ssh, ssh_password=ssh_password,
            ssh_pubkey=ssh_pubkey, ssh_port=ssh_port, ports=port, node=node,
            workspace_path=mount_path, shell=shell, kind=kind, mounts=mounts,
            expose_ssh=expose_ssh, ssh_node_port=ssh_node_port,
        )
    except (WujiError, Exception) as exc:  # noqa: BLE001 - friendly exit
        _die(exc)

    ns = current_namespace(team)
    team_arg = f" --team {team}" if team else ""
    label = "开发机" if kind == "dev" else "任务"
    _echo(f"已提交{label}: {name}  (ns={ns}, queue={queue}, {nodes}x{gpus} GPU, kind={kind})")
    _echo(f"查看状态: volcano status {name}{team_arg}")
    _echo(f"看日志:   volcano logs {name} -f{team_arg}")
    _echo(f"进容器:   volcano exec {name}{team_arg} -- {shell}")

    enable_ssh = ssh or ssh_password or ssh_pubkey or expose_ssh
    port_arg = f" --ssh-port {ssh_port}" if ssh_port != 22 else ""
    gen_pw = result.get("_ssh_generated_password") if isinstance(result, dict) else None

    if enable_ssh and gen_pw:
        _echo("")
        _echo(f"⚠ 未给 --ssh-password/--ssh-pubkey,已自动生成 root 密码(仅此一次显示): {gen_pw}")

    if expose_ssh and nodes > 1:
        worker_nps = (
            result.get("_ssh_worker_node_ports", {}) if isinstance(result, dict) else {}
        )
        _echo("")
        _echo(f"SSH 公网直连就绪 — 每个 worker 一个入口(等 Running+sshd,约 30–60 秒):")
        resolved: dict = {}
        for _ in range(20):  # ~40s
            for i in range(nodes):
                if i in resolved:
                    continue
                try:
                    ep = sdk.worker_ssh_endpoint(name, i, team=team)
                except Exception:  # noqa: BLE001
                    ep = None
                if ep and ep.get("public_ip"):
                    resolved[i] = ep
            if len(resolved) == nodes:
                break
            time.sleep(2)
        for i in range(nodes):
            ep = resolved.get(i)
            if ep and ep.get("public_ip"):
                _echo(
                    f"  worker {i}: ssh -p {ep['node_port']} root@{ep['public_ip']}"
                    f"   (节点 {ep.get('node')})"
                )
            else:
                np = worker_nps.get(i)
                _echo(
                    f"  worker {i}: NodePort {np};公网 IP 待解析,稍后跑 "
                    f"volcano ssh {name} --worker {i}{team_arg}"
                )
        _echo(f"  或走隧道: volcano ssh {name} --worker <i>{team_arg}")
        _echo(
            "  提示:地址已就绪但容器可能还在启动;首次连接被拒属正常,"
            "volcano ssh 会自动等就绪再连,或等 30–60 秒后手动重试。"
        )
    elif expose_ssh:
        node_port = result.get("_ssh_node_port") if isinstance(result, dict) else None
        _echo("")
        _echo("SSH 公网直连就绪(等容器 Running + sshd 起来,约 30–60 秒):")
        _echo("  等待调度并解析节点公网 IP ...")
        endpoint = None
        for _ in range(15):  # ~30s
            try:
                endpoint = sdk.ssh_endpoint(name, team=team)
            except Exception:  # noqa: BLE001
                endpoint = None
            if endpoint and endpoint.get("public_ip"):
                break
            if endpoint and endpoint.get("node"):
                break  # 已调度到节点却拿不到公网 IP:多半缺注解/无 node 读权限,再等无益
            time.sleep(2)
        if endpoint and endpoint.get("public_ip"):
            ip, np = endpoint["public_ip"], endpoint["node_port"]
            _echo(f"  直接连:   ssh -p {np} root@{ip}   (节点 {endpoint.get('node')})")
        elif node_port:
            _echo(
                f"  NodePort {node_port} 已分配,但节点公网 IP 还没解析出来"
                f"(pod 未调度,或节点缺 wuji.io/public-ip 注解/无 node 读权限)。"
            )
            _echo(f"  稍后跑: volcano ssh {name}{team_arg}  # 会自动走公网直连")
        _echo(f"  或仍走隧道: volcano ssh {name}{team_arg}{port_arg}")
        _echo(
            "  提示:地址已就绪但容器可能还在启动;首次连接被拒属正常,"
            "volcano ssh 会自动等就绪再连,或等 30–60 秒后手动重试。"
        )
    elif enable_ssh:
        _echo("")
        _echo("SSH 就绪(root,密码即 --ssh-password;需容器 Running 后稍等 sshd 起来):")
        _echo(f"  一步连:   volcano ssh {name}{team_arg}{port_arg}  # 会自动等容器就绪再连")
        _echo(
            f"  或用 IP:  volcano forward {name} {ssh_port} --local-port 2222{team_arg}  "
            f"# 挂着别关,另开终端: ssh -p 2222 root@127.0.0.1"
        )
    for p in port or []:
        if enable_ssh and p == ssh_port:
            continue  # already covered by the SSH hint above
        _echo(f"转发端口 {p}:  volcano forward {name} {p}{team_arg}  # 本地 http://localhost:{p}")


# --------------------------------------------------------------------------- #
# train / submit (DLC:训练任务)  &  dev (DSW:开发机)
# --------------------------------------------------------------------------- #
@app.command("train")
@app.command("submit", hidden=True)  # 兼容旧名:volcano submit == volcano train
def train(
    command: Optional[List[str]] = typer.Argument(
        None, metavar="[-- CMD ...]",
        help="训练命令,放在 -- 之后,例如: -- python train.py --epochs 3",
    ),
    name: str = typer.Option(..., "--name", "-n", help="任务名(合法 k8s 名)。"),
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="团队=namespace(默认取当前 context)。"
    ),
    queue: Optional[str] = typer.Option(None, "--queue", "-q", help="Volcano 队列;默认投你所属团队队列(ns 的 wuji.io/team 标签),无则 default。"),
    image: str = typer.Option(..., "--image", "-i", help="ACR 镜像完整地址。"),
    gpus: int = typer.Option(8, "--gpus", "-g", help="每个 worker 的 GPU 数。"),
    nodes: int = typer.Option(1, "--nodes", help="worker 数(=minAvailable,多机分布式)。"),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="相对 /workspace 的数据路径。"),
    shared_data: Optional[str] = typer.Option(
        None, "--shared-data", help="相对 /datasets 的共享只读数据集路径。"
    ),
    cpu: str = typer.Option("16", "--cpu", "-c", help="每 worker CPU 限额。"),
    memory: str = typer.Option("64Gi", "--memory", "-m", help="每 worker 内存限额。"),
    ssh: bool = typer.Option(False, "--ssh", help="在容器里起 sshd(root),之后 volcano ssh 进入。"),
    ssh_password: Optional[str] = typer.Option(
        None, "--ssh-password", help="SSH root 密码;会写进 Job env(同 ns 可见)。不给则自动生成。"
    ),
    ssh_pubkey: Optional[str] = typer.Option(
        None, "--ssh-pubkey", help="SSH 公钥(内容或文件路径);给了则关口令登录,公网暴露首选。"
    ),
    ssh_port: int = typer.Option(22, "--ssh-port", help="容器内 sshd 监听端口。"),
    expose_ssh: bool = typer.Option(
        False, "--expose-ssh", help="建 NodePort 把 SSH 暴露到节点弹性公网 IP(含 --ssh)。"
    ),
    ssh_node_port: Optional[int] = typer.Option(
        None, "--ssh-nodeport", help="固定 NodePort(30000-32767);默认自动分配。"
    ),
    port: Optional[List[int]] = typer.Option(
        None, "--port", help="额外暴露的容器端口(可多次)。"
    ),
    node: Optional[str] = typer.Option(None, "--node", help="钉到指定节点调度。"),
    mount: Optional[List[str]] = typer.Option(
        None, "--mount",
        help="额外挂载 PVC:<pvc>:<容器路径>[:ro],可多次(挂 public/oss 等团队卷)。",
    ),
    mount_path: str = typer.Option(
        "/workspace", "--mount-path", help="团队 NAS 在容器里的挂载目录。"
    ),
    shell: str = typer.Option("bash", "--shell", help="包裹命令的 shell(busybox/alpine 用 sh)。"),
    cmd: Optional[str] = typer.Option(None, "--cmd", help="命令整段字符串;与 -- CMD 二选一。"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印 Job YAML,不真提交。"),
) -> None:
    """提交训练任务(DLC:跑完即退出;支持多机分布式)。`volcano submit` 是同义词。"""
    _submit_impl(
        command=command, cmd=cmd, name=name, team=team, queue=queue, image=image,
        gpus=gpus, nodes=nodes, data=data, shared_data=shared_data, cpu=cpu,
        memory=memory, ssh=ssh, ssh_password=ssh_password, ssh_pubkey=ssh_pubkey,
        ssh_port=ssh_port, expose_ssh=expose_ssh, ssh_node_port=ssh_node_port,
        port=port, node=node, mount_path=mount_path, shell=shell, dry_run=dry_run,
        mounts=mount,
        kind="train",
    )


@app.command()
def dev(
    command: Optional[List[str]] = typer.Argument(
        None, metavar="[-- CMD ...]",
        help="容器启动命令(可选);不给则默认长驻 tail -f /dev/null,供你 exec/ssh 进去。",
    ),
    name: str = typer.Option(..., "--name", "-n", help="开发机名(合法 k8s 名)。"),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
    queue: Optional[str] = typer.Option(None, "--queue", "-q", help="Volcano 队列;默认投你所属团队队列(ns 的 wuji.io/team 标签),无则 default。"),
    image: str = typer.Option(..., "--image", "-i", help="ACR 镜像完整地址。"),
    gpus: int = typer.Option(1, "--gpus", "-g", help="GPU 数(开发机默认 1)。"),
    data: Optional[str] = typer.Option(None, "--data", "-d", help="相对 /workspace 的数据路径。"),
    shared_data: Optional[str] = typer.Option(
        None, "--shared-data", help="相对 /datasets 的共享只读数据集路径。"
    ),
    cpu: str = typer.Option("16", "--cpu", "-c", help="CPU 限额。"),
    memory: str = typer.Option("64Gi", "--memory", "-m", help="内存限额。"),
    ssh: bool = typer.Option(False, "--ssh", help="起 sshd(root),之后 volcano ssh 进入。"),
    ssh_password: Optional[str] = typer.Option(
        None, "--ssh-password", help="SSH root 密码;不给则自动生成。"
    ),
    ssh_pubkey: Optional[str] = typer.Option(
        None, "--ssh-pubkey", help="SSH 公钥(内容或文件路径);给了则关口令登录。"
    ),
    ssh_port: int = typer.Option(22, "--ssh-port", help="容器内 sshd 监听端口。"),
    expose_ssh: bool = typer.Option(
        True, "--expose-ssh/--no-expose-ssh",
        help="默认开:起 sshd 并把 SSH 暴露到节点弹性公网 IP,之后 volcano ssh 直连;"
             "不需要公网 SSH 用 --no-expose-ssh 关闭。",
    ),
    ssh_node_port: Optional[int] = typer.Option(
        None, "--ssh-nodeport", help="固定 NodePort(30000-32767);默认自动分配。"
    ),
    port: Optional[List[int]] = typer.Option(None, "--port", help="额外暴露的容器端口(可多次)。"),
    node: Optional[str] = typer.Option(None, "--node", help="钉到指定节点调度。"),
    mount: Optional[List[str]] = typer.Option(
        None, "--mount",
        help="额外挂载 PVC:<pvc>:<容器路径>[:ro],可多次(挂 public/oss 等团队卷)。",
    ),
    mount_path: str = typer.Option("/workspace", "--mount-path", help="NAS 挂载目录。"),
    shell: str = typer.Option("bash", "--shell", help="包裹命令的 shell(busybox/alpine 用 sh)。"),
    cmd: Optional[str] = typer.Option(None, "--cmd", help="启动命令整段字符串。"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只打印 Job YAML,不真提交。"),
) -> None:
    """起一个开发机(DSW:长驻交互容器,用 volcano exec/ssh 进去开发调试)。

    默认开公网 SSH(--expose-ssh):起 sshd 并暴露到节点弹性公网 IP,提交后可直接
    ``volcano ssh <名>`` 连进去(未给密码会自动生成一次性 root 密码)。不需要公网
    SSH 就加 ``--no-expose-ssh``。
    """
    if not command and not cmd:
        cmd = "tail -f /dev/null"  # 默认长驻,否则容器跑完就退、exec/ssh 进不去
    _submit_impl(
        command=command, cmd=cmd, name=name, team=team, queue=queue, image=image,
        gpus=gpus, nodes=1, data=data, shared_data=shared_data, cpu=cpu,
        memory=memory, ssh=ssh, ssh_password=ssh_password, ssh_pubkey=ssh_pubkey,
        ssh_port=ssh_port, expose_ssh=expose_ssh, ssh_node_port=ssh_node_port,
        port=port, node=node, mount_path=mount_path, shell=shell, dry_run=dry_run,
        mounts=mount,
        kind="dev",
    )


# --------------------------------------------------------------------------- #
# save — 把活容器提交成镜像推到 ACR(平台 wuji-saver 代做)
# --------------------------------------------------------------------------- #
@app.command()
def save(
    name: str = typer.Argument(..., help="要保存的开发机 / 任务名。"),
    tag: str = typer.Option(
        ..., "--tag",
        help="镜像短名(如 myenv:v1),自动存到你团队专属的 ACR 空间;不要带 registry 地址。",
    ),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
    worker: int = typer.Option(0, "--worker", "-w", help="第几个 worker(多机);默认 0。"),
    container: str = typer.Option(
        "worker", "--container", help="容器名(volcano 起的任务默认是 worker)。"
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="只提交保存请求,不等待完成(自己稍后查状态)。"
    ),
    timeout: int = typer.Option(1800, "--timeout", help="等待 saver 完成的最长秒数。"),
) -> None:
    """把正在运行的容器提交成镜像并推到 ACR。

    你不需要 docker,也不需要 ACR 凭据:平台的 wuji-saver 组件在容器所在节点用
    containerd 提交并推送(它独占 containerd socket + ACR 写凭据)。你这边只是发一个
    保存请求。改环境后 `volcano save <名> --tag myenv:v1`,下次 `volcano dev/train -i` 直接用。
    """
    try:
        result = sdk.save_image(
            name, tag=tag, team=team, worker=worker, container=container,
            wait=not no_wait, timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        _die(exc)

    image = result["image"]
    ns = current_namespace(team)
    if no_wait:
        _echo(f"已提交保存请求: {result['request']}")
        _echo(f"目标镜像: {image}")
        _echo(
            f"查看进度: kubectl -n {ns} get cm {result['request']} "
            "-o jsonpath='{.data.status}'"
        )
        return
    _echo(f"✅ 已保存并推送: {image}   (节点 {result.get('node') or '?'})")
    _echo("下次直接拿它起环境:")
    _echo(f"  volcano dev   --name <新开发机> -i {image}")
    _echo(f"  volcano train --name <新任务>   -i {image} -- python train.py")


# --------------------------------------------------------------------------- #
# images — 可拉取的镜像清单(平台索引,免 ACR 凭据)
# --------------------------------------------------------------------------- #
@app.command()
def images(
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="只看某团队保存的镜像(基础镜像始终显示)。"
    ),
    mine: bool = typer.Option(
        False, "--mine", help="只看当前团队保存的镜像(= --team 当前 ns)。"
    ),
) -> None:
    """列出可直接拉取的镜像(基础镜像 + 各团队 volcano save 存下的)。

    读的是平台维护的镜像索引(一个共享 ConfigMap),不需要 ACR 凭据 —— ACR 的拉取
    令牌本身没有列目录的权限。想要 ACR 完整权威目录,请管理员用 aliyun cr / 控制台。
    """
    filter_team = current_namespace(team) if mine else team
    try:
        rows = sdk.list_images(team=filter_team)
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    if not rows:
        _echo("镜像索引为空(还没人 volcano save 过,管理员也没种基础镜像)。")
        _echo("管理员可参考 scheduler-platform/saver/README.md 种基础镜像。")
        return
    # 一行一个完整镜像地址(方便直接复制给 -i);不用表格,避免长地址被终端宽度截断。
    base = [r for r in rows if r["kind"] == "base"]
    saved = [r for r in rows if r["kind"] != "base"]
    _echo(f"可拉取镜像:base {len(base)} 个,saved(团队保存){len(saved)} 个")
    if base:
        _echo("")
        _echo("── 基础镜像(base)──")
        for r in base:
            _echo(f"  {r['image']}")
    if saved:
        _echo("")
        _echo("── 团队保存(saved)──")
        for r in saved:
            extra = "  ".join(x for x in [r["team"], r["created"]] if x)
            _echo(f"  {r['image']}" + (f"   [{extra}]" if extra else ""))
    _echo("")
    _echo("用法: 复制上面某一整行完整地址,volcano dev --name <名> -i <地址>  /  volcano train ... -i <地址> -- <命令>")


# --------------------------------------------------------------------------- #
# list / ls
# --------------------------------------------------------------------------- #
@app.command("list")
@app.command("ls", hidden=True)
def list_cmd(
    kind: Optional[str] = typer.Argument(
        None, metavar="[dev|train]",
        help="只看某类:dev(开发机)/ train(训练);不给则全列。",
    ),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
    all_namespaces: bool = typer.Option(
        False, "--all-namespaces", "-A", "-a",
        help="列出所有 namespace 的任务(管理员;需集群级读权限)。",
    ),
) -> None:
    """列出本团队的任务;`volcano list dev` / `volcano list train` 分类查看。

    `-A` 列出全部 namespace(管理员视角,多一列 NS)。
    """
    if kind and kind not in ("dev", "train"):
        _err("list 的过滤参数只能是 dev 或 train。")
        raise typer.Exit(code=1)
    try:
        jobs = sdk.list_jobs(team=team, kind=kind, all_namespaces=all_namespaces)
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    scope = kind or "all"
    if all_namespaces:
        rows = [
            [j.get("namespace", "-"), j["name"], j.get("kind", "-"),
             j["phase"], j["queue"], j["gpus"], j["age"]]
            for j in jobs
        ]
        _print_table(
            f"{scope} vcjobs @ 全部 namespace",
            ["NS", "NAME", "KIND", "PHASE", "QUEUE", "GPUS", "AGE"],
            rows,
        )
    else:
        rows = [
            [j["name"], j.get("kind", "-"), j["phase"], j["queue"], j["gpus"], j["age"]]
            for j in jobs
        ]
        _print_table(
            f"{scope} vcjobs @ {current_namespace(team)}",
            ["NAME", "KIND", "PHASE", "QUEUE", "GPUS", "AGE"],
            rows,
        )
    if not rows:
        _echo(
            "(空)注意:这里只列 volcano 提交的 Volcano Job;"
            "直接用 kubectl 起的原生 Job/Pod 不在此列,用 `kubectl get pods` 查看。"
        )


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
@app.command()
def status(
    name: str = typer.Argument(..., help="任务名。"),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
) -> None:
    """查看某任务的 phase 及其 Pod。"""
    try:
        info = sdk.status(name, team=team)
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    _echo(f"任务 {info['name']}: {info['phase']}")
    _echo(
        f"  资源: GPU {info.get('gpu', 0)}, "
        f"CPU {info.get('cpu_m', 0) / 1000:.0f}核, "
        f"内存 {info.get('mem_gi', 0):.0f}Gi"
    )
    if info.get("message"):
        _echo(f"  说明: {info['message']}")
    rows = [[p["name"], p["phase"], p["node"]] for p in info["pods"]]
    _print_table("Pods", ["POD", "PHASE", "NODE"], rows)


# --------------------------------------------------------------------------- #
# logs
# --------------------------------------------------------------------------- #
@app.command()
def logs(
    name: str = typer.Argument(..., help="任务名。"),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
    follow: bool = typer.Option(False, "-f", "--follow", help="持续跟随日志。"),
    tail: int = typer.Option(200, "--tail", help="非跟随模式下拉取的末尾行数。"),
    worker: Optional[int] = typer.Option(
        None, "--worker", "-w", help="看第几个 worker(多机训练);默认 worker-0/首个 Pod。"
    ),
    pod: Optional[str] = typer.Option(None, "--pod", help="直接指定 Pod 名。"),
    all_workers: bool = typer.Option(
        False, "--all", help="打印所有 worker 的日志(带 [pod] 前缀;不支持 -f)。"
    ),
) -> None:
    """查看任务日志。默认首个 Pod;多机训练用 --worker N / --pod <名> / --all。"""
    if all_workers:
        if follow:
            _err("--all 不支持 -f(多路跟随);去掉 -f,或用 --worker N 跟随单个。")
            raise typer.Exit(code=1)
        try:
            info = sdk.status(name, team=team)
        except Exception as exc:  # noqa: BLE001
            _die(exc)
        pods = info.get("pods", [])
        if not pods:
            _err(f"任务 {name} 还没有 Pod。")
            raise typer.Exit(code=1)
        for p in pods:
            _echo(f"===== [{p['name']}] ({p['phase']}) =====")
            try:
                _echo(sdk.logs(name, team=team, tail=tail, pod=p["name"]))
            except Exception as exc:  # noqa: BLE001
                _err(exc.args[0] if isinstance(exc, WujiError) else str(exc))
        return
    try:
        out = sdk.logs(
            name, team=team, follow=follow, tail=tail, pod=pod, worker=worker
        )
        if follow:
            for line in out:
                _echo(line)
        else:
            _echo(out)
    except Exception as exc:  # noqa: BLE001
        _die(exc)


# --------------------------------------------------------------------------- #
# exec — 进入容器
# --------------------------------------------------------------------------- #
@app.command(name="exec")
def exec_cmd(
    name: str = typer.Argument(..., help="任务名。"),
    command: Optional[List[str]] = typer.Argument(
        None, metavar="[-- CMD ...]", help="要执行的命令,默认 bash。"
    ),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
    pod: Optional[str] = typer.Option(
        None, "--pod", help="指定某个 Pod(多 worker 时);默认第一个 Running 的。"
    ),
    worker: Optional[int] = typer.Option(
        None, "--worker", "-w", help="进第几个 worker(多机训练),等价于 --pod <名>-worker-N。"
    ),
) -> None:
    """进入任务容器(交互式 shell)。

    自动找到该任务的 Pod,底层用 ``kubectl exec -it`` 拿到真正的交互式 TTY,
    所以需要本机装了 kubectl(和 volcano 用同一个 kubeconfig)。
    """
    import shutil
    import subprocess

    if not shutil.which("kubectl"):
        _err("需要本机安装 kubectl(volcano exec 底层用它拿交互式 TTY)。")
        raise typer.Exit(code=1)

    ns = current_namespace(team)
    target_pod = pod or (f"{name}-worker-{worker}" if worker is not None else None)
    if not target_pod:
        try:
            info = sdk.status(name, team=team)
        except Exception as exc:  # noqa: BLE001
            _die(exc)
        if not info["pods"]:
            _err(f"任务 {name} 还没有 Pod(可能还在排队/未调度)。")
            raise typer.Exit(code=1)
        running = [p for p in info["pods"] if p["phase"] == "Running"]
        target_pod = (running or info["pods"])[0]["name"]

    run = list(command) if command else ["bash"]
    _echo(f"进入 {ns}/{target_pod} ...(exit 退出)")
    argv = ["kubectl", "exec", "-it", "-n", ns, target_pod, "--", *run]
    raise typer.Exit(code=subprocess.call(argv))


# --------------------------------------------------------------------------- #
# ssh / forward — 端口转发 & SSH
# --------------------------------------------------------------------------- #
def _wait_pod_running(
    name: str,
    *,
    team: Optional[str],
    pod: Optional[str] = None,
    worker: Optional[int] = None,
    timeout: int = 120,
) -> None:
    """轮询等目标 Pod 到 Running,再返回;超时/取不到状态就放行(不挡着连接)。

    解决"``--expose-ssh`` 地址在提交时就建好并打印,但 pod 还在调度/拉镜像/起 sshd,
    用户去连被拒"的坑:连接前先等容器就绪,并打印进度提示,省得手动重试。
    Ctrl+C 可随时中断等待、直接尝试连接。
    """
    target = pod or (f"{name}-worker-{worker}" if worker is not None else None)
    waited = 0
    announced = False
    # 整个轮询体包一层 KeyboardInterrupt:Ctrl+C 无论落在 sleep 还是 sdk.status 的
    # 网络调用期间,都干净地"中断等待、直接去连",而不是抛裸 traceback。
    try:
        while waited < timeout:
            try:
                info = sdk.status(name, team=team)
            except Exception:  # noqa: BLE001 - 状态拿不到就别挡着,直接放行去连
                return
            pods = info.get("pods", [])
            match = [p for p in pods if p["name"] == target] if target else pods
            if any(p["phase"] == "Running" for p in match):
                if announced:
                    _echo("  ✅ 容器已 Running(sshd 可能还需几秒起来,若首次被拒稍等重试)。")
                return
            if match and not announced:
                _echo(
                    f"⏳ 容器还没就绪(当前 {match[0]['phase']}),等待中 …… "
                    f"最多等 {timeout}s(Ctrl+C 可中断直接连)。"
                )
                announced = True
            time.sleep(3)
            waited += 3
    except KeyboardInterrupt:
        _echo("  (已中断等待,直接尝试连接)")
        return
    if announced:
        _echo(f"  容器 {timeout}s 内仍未 Running;直接尝试连接(可能被拒,请稍后重试)。")


def _first_running_pod(
    name: str, team: Optional[str], pod: Optional[str], worker: Optional[int] = None
):
    """(namespace, pod);pod 未指定时取该任务第一个 Running 的 Pod。

    给了 ``worker`` 而没给 ``pod`` 时,解析为 ``<name>-worker-<n>``。
    """
    ns = current_namespace(team)
    if not pod and worker is not None:
        pod = f"{name}-worker-{worker}"
    if pod:
        return ns, pod
    try:
        info = sdk.status(name, team=team)
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    if not info["pods"]:
        _err(f"任务 {name} 还没有 Pod(可能还在排队)。")
        raise typer.Exit(code=1)
    running = [p for p in info["pods"] if p["phase"] == "Running"]
    return ns, (running or info["pods"])[0]["name"]


@app.command(name="ssh")
def ssh_cmd(
    name: str = typer.Argument(..., help="任务名。"),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
    pod: Optional[str] = typer.Option(None, "--pod", help="指定 Pod;默认第一个 Running。"),
    worker: Optional[int] = typer.Option(
        None, "--worker", "-w", help="进第几个 worker(多机训练)。"
    ),
    ssh_port: int = typer.Option(22, "--ssh-port", help="容器内 sshd 端口(须与提交时一致)。"),
    local_port: int = typer.Option(2222, "--local-port", help="本地映射端口。"),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="不等容器就绪,直接尝试连接(默认会先等 Pod Running)。"
    ),
) -> None:
    """SSH 进任务容器(root)。需提交时开了 --ssh。

    边缘节点无公网 IP,故用 ``kubectl port-forward`` 把容器 sshd 端口映射到本地再 ssh。
    需本机装 kubectl + ssh。

    默认会先等目标 Pod 到 Running 再连(容器还在拉镜像/调度时,提交时打印的 SSH 地址
    连过去会被拒——这里替你等好);急着连或想自己重试用 ``--no-wait``。
    """
    import shutil
    import subprocess
    import time

    if not shutil.which("ssh"):
        _err("需要本机安装 ssh。")
        raise typer.Exit(code=1)

    # 连接前先等容器就绪(除非 --no-wait):地址提交即建、容器却可能没起,先等再连省得手动重试。
    if not no_wait:
        _wait_pod_running(name, team=team, pod=pod, worker=worker)

    # 若任务用 --expose-ssh 提交,直接公网连,免 port-forward:
    #   指定 --worker i → 该 worker 的独立公网入口(多机 per-worker NodePort);
    #   不指定 → 任务级入口(单机);指定 --pod 时走隧道(无对应公网口)。
    try:
        if pod:
            endpoint = None
        elif worker is not None:
            endpoint = sdk.worker_ssh_endpoint(name, worker, team=team)
        else:
            endpoint = sdk.ssh_endpoint(name, team=team)
    except Exception:  # noqa: BLE001
        endpoint = None
    if endpoint and endpoint.get("public_ip") and endpoint.get("node_port"):
        ip, np = endpoint["public_ip"], endpoint["node_port"]
        _echo(f"公网直连 root@{ip}:{np}(节点 {endpoint.get('node')})...")
        # 公网路径:用 TOFU(accept-new)并记住主机密钥,不要 UserKnownHostsFile=/dev/null——
        # 那会让每次都盲信任意服务端,root+口令穿公网时可被中间人截获。
        code = subprocess.call([
            "ssh", "-p", str(np),
            "-o", "StrictHostKeyChecking=accept-new",
            f"root@{ip}",
        ])
        raise typer.Exit(code=code)

    if not shutil.which("kubectl"):
        _err("需要本机安装 kubectl(无 --expose-ssh 时走 port-forward)。")
        raise typer.Exit(code=1)
    ns, target = _first_running_pod(name, team, pod, worker)
    pf = subprocess.Popen(
        ["kubectl", "port-forward", "-n", ns, target, f"{local_port}:{ssh_port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2)
        _echo(f"port-forward {ns}/{target}:{ssh_port} → localhost:{local_port};连接 root@ ...")
        code = subprocess.call([
            "ssh", "-p", str(local_port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "root@127.0.0.1",
        ])
        raise typer.Exit(code=code)
    finally:
        pf.terminate()


@app.command()
def forward(
    name: str = typer.Argument(..., help="任务名。"),
    port: int = typer.Argument(..., help="容器端口(如 6006 TensorBoard、8265 Ray Dashboard)。"),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
    pod: Optional[str] = typer.Option(None, "--pod", help="指定 Pod;默认第一个 Running。"),
    worker: Optional[int] = typer.Option(None, "--worker", "-w", help="第几个 worker(多机)。"),
    local_port: Optional[int] = typer.Option(None, "--local-port", help="本地端口(默认同容器端口)。"),
) -> None:
    """把容器端口转发到本地(访问 TensorBoard / Ray Dashboard 等)。前台运行,Ctrl+C 结束。"""
    import shutil
    import subprocess

    if not shutil.which("kubectl"):
        _err("需要本机安装 kubectl。")
        raise typer.Exit(code=1)
    ns, target = _first_running_pod(name, team, pod, worker)
    lp = local_port or port
    _echo(f"转发 {ns}/{target}:{port} → localhost:{lp}(浏览器开 http://localhost:{lp};Ctrl+C 结束)")
    raise typer.Exit(
        code=subprocess.call(
            ["kubectl", "port-forward", "-n", ns, target, f"{lp}:{port}"]
        )
    )


# --------------------------------------------------------------------------- #
# kill
# --------------------------------------------------------------------------- #
@app.command()
def kill(
    name: str = typer.Argument(..., help="任务名。"),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
) -> None:
    """删除某 Volcano 任务。"""
    try:
        sdk.kill(name, team=team)
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    _echo(f"已删除任务: {name}")


# --------------------------------------------------------------------------- #
# queue
# --------------------------------------------------------------------------- #
def _queue_row(q: dict) -> List[str]:
    """One queue → [队列, 状态, GPU 用/配/顶, CPU核(已用), 内存Gi(已用)]."""
    a, d, c = q["allocated"], q["deserved"], q["capability"]
    return [
        q["name"], q["state"],
        f"{a['gpu']}/{d['gpu']}/{c['gpu']}",
        f"{a['cpu_m'] / 1000:.0f}",
        f"{a['mem_gi']:.0f}",
    ]


@app.command("volumes")
@app.command("vols", hidden=True)
def volumes(
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
) -> None:
    """列出本团队可挂载的 PVC(用 `--mount <pvc>:<路径>` 挂进容器)。"""
    try:
        pvcs = sdk.list_pvcs(team=team)
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    rows = [
        [p["name"], p["status"], p["capacity"], p["storageclass"], p["volume"]]
        for p in pvcs
    ]
    _print_table(
        f"可挂载 PVC @ {current_namespace(team)}",
        ["NAME", "STATUS", "容量", "STORAGECLASS", "PV"],
        rows,
    )
    _echo(
        "挂载:volcano dev/train --mount <NAME>:/容器路径[:ro]  "
        "(默认已挂 <team>-nas → /workspace,无需手动挂)"
    )


@app.command()
def queue() -> None:
    """列出各 Volcano 队列的 GPU/CPU/内存 配额与用量。"""
    try:
        queues = sdk.list_queues()
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    _print_table(
        "Volcano 队列  (GPU: 已用/配额deserved/上限cap;CPU核、内存Gi 为已用)",
        ["队列", "状态", "GPU 用/配/顶", "CPU核", "内存Gi"],
        [_queue_row(q) for q in queues],
    )
    _echo(
        "提示:这里的「已用」只统计走 Volcano 调度(schedulerName: volcano)的任务;"
        "直接用 kubectl/default-scheduler 起的任务不计入队列账。看节点真实空闲用 `volcano top`。"
    )


@app.command()
def top() -> None:
    """看资源:每个 GPU 节点的 GPU/CPU/内存(已用/总)+ 各队列用量。"""
    try:
        nodes = sdk.top_nodes()
        queues = sdk.list_queues()
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    tg = sum(n["gpu_used"] for n in nodes)
    tc = sum(n["gpu_cap"] for n in nodes)
    _print_table(
        f"GPU 节点资源(已用/总)   GPU 合计 {tg}/{tc}",
        ["节点", "GPU", "CPU核", "内存Gi"],
        [
            [
                n["node"],
                f"{n['gpu_used']}/{n['gpu_cap']}",
                f"{n['cpu_used_m'] / 1000:.0f}/{n['cpu_cap_m'] / 1000:.0f}",
                f"{n['mem_used_gi']:.0f}/{n['mem_cap_gi']:.0f}",
            ]
            for n in nodes
        ],
    )
    _print_table(
        "各队列用量  (GPU: 已用/配额/上限;CPU核、内存Gi 为已用)",
        ["队列", "状态", "GPU 用/配/顶", "CPU核", "内存Gi"],
        [_queue_row(q) for q in queues],
    )


@app.command()
def usage(
    team: Optional[str] = typer.Option(
        None, "--team", "-t", help="只看某人/团队(namespace);默认列所有在占 GPU 的。"
    ),
) -> None:
    """看某人/团队(namespace)占用的 GPU/CPU/内存。CPU 为逻辑核(物理核≈半数)。"""
    try:
        rows = sdk.usage_by_namespace(team=team)
    except Exception as exc:  # noqa: BLE001
        _die(exc)
    _print_table(
        f"占用 — {team}" if team else "各命名空间占用(在占 GPU 的,按 GPU 降序)",
        ["命名空间", "GPU", "CPU核", "内存Gi", "Pods"],
        [
            [
                r["namespace"], int(r["gpu"]),
                f"{r['cpu_m'] / 1000:.0f}", f"{r['mem_gi']:.0f}", int(r["pods"]),
            ]
            for r in rows
        ],
    )


# --------------------------------------------------------------------------- #
# data ls
# --------------------------------------------------------------------------- #
@data_app.command("ls")
def data_ls(
    path: str = typer.Argument("", help="相对挂载点的子路径(默认根)。"),
    team: Optional[str] = typer.Option(None, "--team", "-t", help="团队=namespace。"),
    shared: bool = typer.Option(
        False, "--shared", help="列共享只读数据集(/datasets)而非团队 NAS(/workspace)。"
    ),
    shared_dataset_pvc: str = typer.Option(
        "shared-datasets", "--shared-pvc", help="共享数据集 PVC 名。"
    ),
    timeout: int = typer.Option(60, "--timeout", help="helper Pod 等待秒数。"),
) -> None:
    """列出 NAS / 共享数据集内容。

    CLI 在集群外无法直接读 NFS,因此这里起一个短命 helper Pod 挂上对应 PVC
    跑 ``ls -la``,再取其日志返回。helper Pod 结束后会被删除。
    """
    ns = current_namespace(team)
    if shared:
        pvc, mount = shared_dataset_pvc, "/datasets"
    else:
        pvc, mount = f"{ns}-nas", "/workspace"
    target = f"{mount}/{path.strip().lstrip('/')}" if path.strip() else mount

    try:
        core, _, _ = load_clients()
    except Exception as exc:  # noqa: BLE001
        _die(exc)

    pod_name = f"wuji-ls-{int(time.time())}"
    pod_body = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "labels": {"app.kubernetes.io/managed-by": "wuji"}},
        "spec": {
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "ls",
                    # 用边缘节点已预热的 ubuntu:24.04(image-prepull 常驻),秒起免冷拉;
                    # 别用跨区镜像,呼和浩特边缘冷拉易超时。
                    "image": "ubuntu:24.04",
                    "command": ["sh", "-c", f"ls -la {target}"],
                    "volumeMounts": [{"name": "vol", "mountPath": mount}],
                }
            ],
            "volumes": [
                {"name": "vol", "persistentVolumeClaim": {"claimName": pvc}}
            ],
        },
    }

    from kubernetes.client.exceptions import ApiException

    _echo(f"起 helper Pod 挂 {pvc} 读 {target} ...")
    try:
        core.create_namespaced_pod(namespace=ns, body=pod_body)
    except ApiException as exc:
        _die(WujiError(humanize_api_exception(exc)))

    try:
        deadline = time.time() + timeout
        phase = "Pending"
        while time.time() < deadline:
            pod = core.read_namespaced_pod(name=pod_name, namespace=ns)
            phase = pod.status.phase if pod.status else "Pending"
            if phase in ("Succeeded", "Failed"):
                break
            time.sleep(1)
        if phase == "Pending":
            _err(
                f"helper Pod {timeout}s 内没起来(仍 Pending):多半是镜像没拉到或没排上"
                f"节点,而不是路径问题。可加大 --timeout 重试。"
            )
            return
        out = ""
        try:
            out = core.read_namespaced_pod_log(name=pod_name, namespace=ns)
        except ApiException:
            pass
        if out.strip():
            _echo(out)
        if phase == "Failed":
            # ls 找不到路径会打印 "No such file"/"cannot access";其余算容器/环境错误。
            if "No such file" in out or "cannot access" in out:
                _err(f"路径不存在: {target}")
            else:
                _err("helper Pod 执行失败(phase=Failed);以上为容器输出(若为空,多为镜像/环境问题)。")
    except ApiException as exc:
        _err(humanize_api_exception(exc))
    finally:
        try:
            core.delete_namespaced_pod(name=pod_name, namespace=ns)
        except ApiException:
            pass


def main() -> None:
    """Console-script entry point (``volcano = volcano.cli:main``).

    顶层兜底:把 WujiError(如未指定团队/namespace 解析失败)呈现为一行友好提示,
    而不是裸 traceback —— 覆盖那些在命令 try 块之外解析 namespace 的路径。
    """
    try:
        app()
    except WujiError as exc:
        _err(str(exc))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
