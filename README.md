# volcano — 训练 / 开发机 CLI + SDK

把训练任务或交互式开发机提交成 **Volcano Job**,跑在 ACK Edge GPU 集群上。既是命令行工具
(`volcano ...`),也是 Python SDK(`import volcano`)。

> `vc` 是 `volcano` 的等价短别名,下文所有 `volcano ...` 都可写成 `vc ...`(如 `vc ls`、`vc top`)。

## 它替你处理了什么

- `schedulerName: volcano` + `queue` + gang 调度(`minAvailable = nodes`)。
- 三个标准挂载:
  - 团队 NAS PVC `<team>-nas` → `/workspace`(读写);
  - 共享只读数据集 PVC(默认 `shared-datasets`)→ `/datasets`(只读);
  - `dshm` = 64Gi 内存盘 → `/dev/shm`(PyTorch 多卡 / NCCL 必需)。
- **不写 imagePullSecret**:命名空间 `default` SA 已绑 `acr-cred`。
- 数据路径解析:`--data foo` → 容器内 `/workspace/foo`,`--shared-data bar` →
  `/datasets/bar`;解析出的绝对路径会追加到命令(`--data <abs>`)并注入环境变量
  `WUJI_DATA` / `WUJI_SHARED_DATA`。
- 多机(`--nodes N`)自动配 rendezvous:`MASTER_ADDR` / `MASTER_PORT=29500` /
  `NNODES` / `NODE_RANK`,`torchrun` 开箱即用。

## 安装

前提:Python 3.9+。三选一。

**方式 A — 从 GitHub 直接安装(推荐,零额外基建)**

```bash
pip install "git+https://github.com/ZH-Kinger/volcano-cli.git"
# 想要漂亮的彩色表格输出(volcano top/queue/list 更好看),装 rich extra:
pip install "volcano[rich] @ git+https://github.com/ZH-Kinger/volcano-cli.git"
```

> 国内网络直连 GitHub 可能较慢/不稳,若卡住可挂代理或改用方式 B 离线 wheel。

**方式 B — 用 wheel 离线/内网安装**

```bash
# 管理员把 volcano-0.4.0-py3-none-any.whl 发给你后:
pip install volcano-0.4.0-py3-none-any.whl
pip install "volcano-0.4.0-py3-none-any.whl[rich]"
```

**方式 C — 源码开发安装(想改 CLI 的人)**

```bash
git clone https://github.com/ZH-Kinger/volcano-cli.git
cd volcano-cli
pip install -e ".[rich]"
```

**验证安装成功**

```bash
volcano --help        # 能列出 login/train/dev/list/logs/save/images 等命令即 OK
vc --help             # vc 是等价短别名
```

### 登录(拿到 kubeconfig 后的第一步)

管理员开通团队时给你一份 kubeconfig,用 `volcano login` 写到默认位置,之后 `volcano` / `kubectl`
免设 `KUBECONFIG` 直接用:

```bash
volcano login -f 我的.kubeconfig     # 从文件读
volcano login                        # 或直接运行,粘贴内容(结束:Unix Ctrl-D / Windows Ctrl-Z 回车)
```

写到 `~/.kube/config`(`--dest` 改路径);已存在会先自动备份成 `config.bak.<时间戳>`(`--force` 跳过备份直接覆盖)。
成功后打印当前 context 和默认 namespace。

### 安装后的前提(不满足命令会失败)

1. 本机要有一份**团队作用域的 kubeconfig**(管理员用 `add-team.sh` 开通团队时下发);或每条命令都带 `--team <团队名>`。**不要用没设默认 namespace 的 admin kubeconfig**——否则任务落到 `default` namespace,而 `default` 没有 NAS PVC,pod 会 FailedMount 卡住。
2. 本机需装 `kubectl` 和 `ssh`(`volcano exec` / `ssh` / `forward` 依赖它们)。
3. 团队须已被管理员开通(存在 `<team>-nas` PVC),否则挂载失败。
4. apiserver 公网端点可直连(集群已放开)。

## 认证(自动双模)

无需配置,`volcano` 会自动选择:

1. **集群内**(notebook / Pod):用挂载的 ServiceAccount(`load_incluster_config`)。
2. **笔记本 / 工作站**:用本地 kubeconfig(`load_kube_config`,读 `~/.kube/config`
   或 `$KUBECONFIG`)。

namespace 解析顺序:`--team` > 当前 kubeconfig context 的 namespace > 集群内 SA
namespace > `default`。

## 命令一览

> 标 `-t` 的命令都支持 `--team <名>`(短写 `-t`)指定团队/namespace;不给则用当前 context。

| 命令 | 作用 |
|---|---|
| `volcano login [-f 文件]` | 把 kubeconfig 写到 `~/.kube/config`(拿到 kubeconfig 后第一步) |
| `volcano train ...` | 提交训练任务(DLC:跑完即退出;支持多机;`volcano submit` 是隐藏别名) |
| `volcano dev ...` | 起开发机(DSW:默认 1 卡 + 长驻 `tail -f /dev/null`,进去交互调试) |
| `volcano save <名> --tag <镜像>` | 把活容器提交成镜像推 ACR(平台代做,用户零 docker/凭据) |
| `volcano images [--mine\|-t]` | 列可直接拉取的镜像(base 基础镜像 + saved 团队保存) |
| `volcano list [dev\|train]` | 列本团队任务(别名 `volcano ls`;`-A` 列全部 namespace) |
| `volcano status <名>` | 看某任务 phase + 各 Pod + 资源 |
| `volcano logs <名> [-f]` | 看日志(`--worker N` / `--pod <名>` / `--all`) |
| `volcano exec <名> [-- CMD]` | 进容器交互 shell(默认 bash;需本机 kubectl) |
| `volcano ssh <名>` | SSH 进容器(root;需提交时开 `--ssh`;隧道或公网直连) |
| `volcano forward <名> <端口>` | 转发容器端口到本地(TensorBoard 6006 / Ray 8265 等) |
| `volcano kill <名>` | 删任务(连带 `--expose-ssh` 建的 NodePort Service) |
| `volcano top` | 每个 GPU 节点 GPU/CPU/内存(已用/总)+ 各队列用量 |
| `volcano usage [-t]` | 某人/团队(namespace)占用的 GPU/CPU/内存 |
| `volcano queue` | 各 Volcano 队列的 GPU/CPU/内存 配额与用量 |
| `volcano volumes` | 列本团队可挂载的 PVC(别名 `volcano vols`),配合 `--mount` 用 |
| `volcano data ls [子路径] [--shared]` | 列 NAS(`/workspace`)或共享数据集(`/datasets`) |

`volcano list -A`(或 `-a`)列出**所有 namespace** 的任务,多一列 NS;admin 和普通用户都能用(只读全局可见)。

### 短参速查

| 短 | 长 | 含义 |
|---|---|---|
| `-n` | `--name` | 任务名 |
| `-t` | `--team` | 团队 = namespace |
| `-i` | `--image` | 镜像地址 |
| `-g` | `--gpus` | 每 worker GPU 数 |
| `-q` | `--queue` | Volcano 队列 |
| `-c` | `--cpu` | CPU 限额 |
| `-m` | `--memory` | 内存限额 |
| `-d` | `--data` | 相对 `/workspace` 的数据路径 |
| `-w` | `--worker` | 第几个 worker(`logs`/`exec`/`ssh`/`forward`/`save`) |
| `-f` | `--follow` / `--file` | `logs` 跟随 / `login` 读文件 |
| `-A` `-a` | `--all-namespaces` | `list` 列全部 namespace |
| （无） | `--mount` | 额外挂载 PVC:`<pvc>:<容器路径>[:ro]`,可多次 |

## train vs dev

- **`volcano train`**(旧名 `volcano submit`,仍可用):训练任务,命令跑完即退出;默认 8 卡;
  支持 `--nodes N` 多机分布式。
- **`volcano dev`**:开发机,默认 **1 卡**,不给命令则默认 `tail -f /dev/null` 长驻,用
  `volcano exec` / `volcano ssh` 进去开发调试。单机(不接受 `--nodes`)。

两者共享几乎全部参数(见下)。

```bash
# 训练:单机 8 卡(命令放在 -- 之后;不写 --queue 默认进 default 池)
volcano train --name my-run \
  --image wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com/wuji-rl/<你的镜像>:<tag> \
  --gpus 8 --data mydata/ds -- python train.py --epochs 3

# 训练:多机分布式(2 台 × 8 卡,gang 一起起)
volcano train --name ddp-run --nodes 2 --gpus 8 \
  --image <ACR 镜像> -- torchrun --nproc_per_node=8 train.py

# 开发机:1 卡长驻,进去调试
volcano dev --name mydev --image <ACR 镜像> --ssh
volcano exec mydev             # 进容器
```

### 提交参数(train / dev 通用)

| 参数(短) | 说明 | 默认 |
|---|---|---|
| `--name` | 任务名(合法 k8s 名) | 必填 |
| `--image` `-i` | ACR 镜像完整地址 | 必填 |
| `--team` `-t` | 团队 = namespace | 当前 context |
| `--queue` `-q` | Volcano 队列(`default` 通用池,或 `wuji-queue-1..5` 限额队列各 8 卡) | `default` |
| `--gpus` `-g` | 每 worker GPU 数 | train `8` / dev `1` |
| `--nodes` | worker 数 = gang minAvailable(仅 `train`) | `1` |
| `--cpu` `-c` / `--memory` `-m` | 每 worker CPU / 内存限额 | `16` / `64Gi` |
| `--data` | 相对 `/workspace` 的数据路径 | — |
| `--shared-data` | 相对 `/datasets` 的共享只读数据集路径 | — |
| `--node` | 钉到指定节点(如 `wuji-3` 大内存机) | — |
| `--mount-path` | 团队 NAS 在容器里的挂载目录(不能是 `/`) | `/workspace` |
| `--mount` | 额外挂载 PVC:`<pvc>:<容器路径>[:ro]`,可多次(挂 `public`/`oss` 等团队卷) | — |
| `--shell` | 包裹命令的 shell(busybox/alpine 用 `sh`) | `bash` |
| `--cmd` | 整段命令字符串(与 `-- CMD` 二选一) | — |
| `--port` | 额外暴露的容器端口,可多次 | — |
| `--dry-run` | 只打印 Job YAML,不真提交 | 关 |

### SSH 参数

| 参数 | 说明 |
|---|---|
| `--ssh` | 在容器里起 sshd(root),之后 `volcano ssh` 进入 |
| `--ssh-password` | root 密码(写进 Job env,同 ns 可见)。**不给则自动生成随机密码**,提交后打印一次 |
| `--ssh-pubkey` | SSH 公钥(**文件路径或公钥内容**);给了则关口令登录,公网暴露首选 |
| `--ssh-port` | 容器内 sshd 端口 | 
| `--expose-ssh` | 建 NodePort 把 SSH 暴露到节点弹性公网 IP,可 `ssh` 单跳直连(隐含 `--ssh`) |
| `--ssh-nodeport` | 固定 NodePort(30000-32767);默认自动分配 |

`--ssh-port` 默认 22。`--expose-ssh` 多机(`--nodes>1`)时**每个 worker 一个独立公网入口**,
提交后逐个打印 `ssh -p <NodePort> root@<公网IP>`。

### 进容器的三种方式

| 方式 | 走哪条路 | 受 RBAC 保护 | 适合 |
|---|---|---|---|
| `volcano exec <名>` | apiserver `exec` | 是 | 最省事,拿个交互 shell |
| `volcano ssh <名>`(默认) | `kubectl port-forward` 隧道 → `127.0.0.1` | 是 | VS Code Remote、`scp`/`rsync` |
| `volcano ssh <名>` + 提交时 `--expose-ssh` | NodePort → 节点弹性公网 IP,单跳直连 | **否** | 延迟/吞吐最好、大文件;只认 SSH 口令/密钥 |

> `--expose-ssh` 的 NodePort **绕过 apiserver 和 RBAC**,进去是 root 且能读写团队 NAS。安全组当前对
> `0.0.0.0/0` 开放 `30000-32767`;优先用密钥、用完 `volcano kill`。前提:目标节点已打
> `wuji.io/public-ip` 注解 + 安全组放行 NodePort 段。

## 额外挂载 PVC(`--mount` / `volcano volumes`)

默认已把团队 `<team>-nas` 挂到 `/workspace`,一般够用。要再挂别的团队卷(`public`、`oss-workspace` 等)时用 `--mount`(train/dev 都支持,可多次):

```bash
volcano volumes                                   # 先看本团队能挂哪些 PVC(别名 vc vols)
volcano dev -n d1 -i ubuntu:24.04 \
  --mount public:/public --mount oss-workspace:/oss:ro
```

- 格式 `<pvc>:<容器路径>[:ro]`,第三段 `ro` 只读、`rw`/省略为读写;PVC 名从 `volcano volumes` 里挑。
- 容器路径须绝对,且不能撞内置挂载点 `/workspace`、`/datasets`、`/dev/shm`(否则报错)。
- `volcano volumes`(`vols`)列出本团队 namespace 下所有 PVC:NAME / STATUS / 容量 / STORAGECLASS / PV。

## 存储:数据放哪

统一用 NAS，别把大文件堆在容器根目录:

- `/workspace`(NAS，持久):数据、代码、checkpoint 都放这，跨任务/重启保留、多机共享同一份。默认标准位置。
- 容器根 `/`：节点本地系统盘，临时且多任务共享；写大文件会写爆系统盘 → DiskPressure 驱逐，连累同节点其他任务。只放临时小文件。
- 本地 NVMe 快盘:平台默认不建 PVC，一般不用。

## 保存 / 复用镜像

`volcano dev` 进去改完环境(装 apt 包、编 CUDA 组件等),一条 `volcano save` 把当前容器提交成镜像推
ACR,下次 `volcano dev/train -i` 直接用。你**不需要 docker,也不需要 ACR 凭据**——平台的
`wuji-saver` 组件在容器所在节点用 containerd 代做 commit + push,你只发一个保存请求。

```bash
volcano save mydev --tag myenv:v1     # 裸名自动补成 ACR 地址;也可给完整 registry 地址
volcano images                        # 看到 myenv:v1(kind=saved)
volcano train --name run1 -i wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com/wuji-rl/myenv:v1 -- python train.py
```

> 纯 Python 依赖建议装到 `/workspace/venv`(NAS,持久),那样根本不用存镜像。`volcano save` 主要用于
> **系统级**改动。

### `volcano save` 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--tag` | 目标镜像。裸名 `myenv:v1` 自动补成 `<ACR>/wuji-rl/myenv:v1`;缺 `:tag` 补 `:latest` | 必填 |
| `--worker` | 第几个 worker(多机);pod `<名>-worker-<n>` | `0` |
| `--container` | 容器名(volcano 起的任务默认 `worker`) | `worker` |
| `--no-wait` | 只提交请求不等待完成(自己稍后查 ConfigMap 状态) | 关 |
| `--timeout` | 等 saver 完成的最长秒数 | `1800` |

目标 pod 必须存在且 `Running`,否则立即报错。安全模型与部署见
[`scheduler-platform/saver/README.md`](../saver/README.md)。

### `volcano images`

列出可直接拉取的镜像,读的是平台维护的镜像索引(共享 ConfigMap `wuji-public/wuji-images`),
**不需要 ACR 凭据**(ACR 拉取令牌本身没有列目录权限)。`--mine` 只看当前团队保存的,`-t <团队>`
看指定团队的(base 基础镜像始终显示)。ACR 完整权威目录仍走 `aliyun cr` / 控制台(管理员侧)。

## SDK 用法

```python
import volcano

volcano.submit(
    name="my-run", team="wuji-rl", queue="default",
    image="wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com/wuji-rl/<你的镜像>:<tag>",
    gpus=8, nodes=1,
    command="python train.py",          # 或 ["python", "train.py"]
    data="mydata/ds",                    # → /workspace/mydata/ds, env WUJI_DATA
)

for j in volcano.list_jobs(team="wuji-rl"):
    print(j["name"], j["phase"], j["gpus"])

print(volcano.status("my-run", team="wuji-rl"))
print(volcano.logs("my-run", team="wuji-rl", tail=100))
volcano.kill("my-run", team="wuji-rl")

# 保存活容器为镜像(平台 wuji-saver 代做):
volcano.save_image("mydev", tag="myenv:v1", team="wuji-rl")   # 返回 {request, image, status, node, message}

# 列可拉取镜像索引:
for r in volcano.list_images(team="wuji-rl"):
    print(r["image"], r["kind"], r["team"])

# 只生成 manifest 不提交:
from volcano import build_volcano_job
manifest = build_volcano_job(name="x", team="wuji-rl", queue="default",
                             image="<镜像>", gpus=8, command="python train.py")
```

## 备注

- 队列由管理员用 `scheduling.volcano.sh/v1beta1 Queue` 预建:`default`(通用池,不写 `--queue`
  默认落这)+ `wuji-queue-1..5`(各限 8 卡,用 `-q wuji-queue-N` 走限额)。旧的 `shared` / `pilot`
  已废弃删除。
- `volcano data ls` 在集群外无法直接读 NFS,会起一个短命 helper Pod 挂对应 PVC 跑 `ls`,取日志后删掉。
- 没连集群 / 没 kubeconfig 时,命令给出清晰的中文报错而不是抛栈。
