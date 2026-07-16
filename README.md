# wuji — 训练 / 开发机 CLI + SDK

把训练任务或交互式开发机提交成 **Volcano Job**,跑在 ACK Edge GPU 集群上。既是命令行工具
(`wuji ...`),也是 Python SDK(`import wuji`)。

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
# 想要漂亮的彩色表格输出(wuji top/queue/list 更好看),装 rich extra:
pip install "wuji[rich] @ git+https://github.com/ZH-Kinger/volcano-cli.git"
```

> 国内网络直连 GitHub 可能较慢/不稳,若卡住可挂代理或改用方式 B 离线 wheel。

**方式 B — 用 wheel 离线/内网安装**

```bash
# 管理员把 wuji-0.1.0-py3-none-any.whl 发给你后:
pip install wuji-0.1.0-py3-none-any.whl
pip install "wuji-0.1.0-py3-none-any.whl[rich]"
```

**方式 C — 源码开发安装(想改 CLI 的人)**

```bash
git clone https://github.com/ZH-Kinger/volcano-cli.git
cd volcano-cli
pip install -e ".[rich]"
```

**验证安装成功**

```bash
wuji --help        # 能列出 submit/train/dev/list/logs/save/images 等命令即 OK
```

### 安装后的前提(不满足命令会失败)

1. 本机要有一份**团队作用域的 kubeconfig**(管理员用 `add-team.sh` 开通团队时下发);或每条命令都带 `--team <团队名>`。**不要用没设默认 namespace 的 admin kubeconfig**——否则任务落到 `default` namespace,而 `default` 没有 NAS PVC,pod 会 FailedMount 卡住。
2. 本机需装 `kubectl` 和 `ssh`(`wuji exec` / `ssh` / `forward` 依赖它们)。
3. 团队须已被管理员开通(存在 `<team>-nas` PVC),否则挂载失败。
4. apiserver 公网端点可直连(集群已放开)。

## 认证(自动双模)

无需配置,`wuji` 会自动选择:

1. **集群内**(notebook / Pod):用挂载的 ServiceAccount(`load_incluster_config`)。
2. **笔记本 / 工作站**:用本地 kubeconfig(`load_kube_config`,读 `~/.kube/config`
   或 `$KUBECONFIG`)。

namespace 解析顺序:`--team` > 当前 kubeconfig context 的 namespace > 集群内 SA
namespace > `default`。

## 命令一览

> 标 `-t` 的命令都支持 `--team <名>`(短写 `-t`)指定团队/namespace;不给则用当前 context。

| 命令 | 作用 |
|---|---|
| `wuji train ...` | 提交训练任务(DLC:跑完即退出;支持多机;`wuji submit` 是隐藏别名) |
| `wuji dev ...` | 起开发机(DSW:默认 1 卡 + 长驻 `tail -f /dev/null`,进去交互调试) |
| `wuji save <名> --tag <镜像>` | 把活容器提交成镜像推 ACR(平台代做,用户零 docker/凭据) |
| `wuji images [--mine\|-t]` | 列可直接拉取的镜像(base 基础镜像 + saved 团队保存) |
| `wuji list [dev\|train]` | 列本团队任务(别名 `wuji ls`) |
| `wuji status <名>` | 看某任务 phase + 各 Pod + 资源 |
| `wuji logs <名> [-f]` | 看日志(`--worker N` / `--pod <名>` / `--all`) |
| `wuji exec <名> [-- CMD]` | 进容器交互 shell(默认 bash;需本机 kubectl) |
| `wuji ssh <名>` | SSH 进容器(root;需提交时开 `--ssh`;隧道或公网直连) |
| `wuji forward <名> <端口>` | 转发容器端口到本地(TensorBoard 6006 / Ray 8265 等) |
| `wuji kill <名>` | 删任务(连带 `--expose-ssh` 建的 NodePort Service) |
| `wuji top` | 每个 GPU 节点 GPU/CPU/内存(已用/总)+ 各队列用量 |
| `wuji usage [-t]` | 某人/团队(namespace)占用的 GPU/CPU/内存 |
| `wuji queue` | 各 Volcano 队列的 GPU/CPU/内存 配额与用量 |
| `wuji data ls [子路径] [--shared]` | 列 NAS(`/workspace`)或共享数据集(`/datasets`) |

## train vs dev

- **`wuji train`**(旧名 `wuji submit`,仍可用):训练任务,命令跑完即退出;默认 8 卡;
  支持 `--nodes N` 多机分布式。
- **`wuji dev`**:开发机,默认 **1 卡**,不给命令则默认 `tail -f /dev/null` 长驻,用
  `wuji exec` / `wuji ssh` 进去开发调试。单机(不接受 `--nodes`)。

两者共享几乎全部参数(见下)。

```bash
# 训练:单机 8 卡(命令放在 -- 之后)
wuji train --name my-run --queue shared \
  --image wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com/wuji-rl/<你的镜像>:<tag> \
  --gpus 8 --data mydata/ds -- python train.py --epochs 3

# 训练:多机分布式(2 台 × 8 卡,gang 一起起)
wuji train --name ddp-run --nodes 2 --gpus 8 \
  --image <ACR 镜像> -- torchrun --nproc_per_node=8 train.py

# 开发机:1 卡长驻,进去调试
wuji dev --name mydev --image <ACR 镜像> --ssh
wuji exec mydev             # 进容器
```

### 提交参数(train / dev 通用)

| 参数(短) | 说明 | 默认 |
|---|---|---|
| `--name` | 任务名(合法 k8s 名) | 必填 |
| `--image` `-i` | ACR 镜像完整地址 | 必填 |
| `--team` `-t` | 团队 = namespace | 当前 context |
| `--queue` `-q` | Volcano 队列 | `shared` |
| `--gpus` `-g` | 每 worker GPU 数 | train `8` / dev `1` |
| `--nodes` | worker 数 = gang minAvailable(仅 `train`) | `1` |
| `--cpu` `-c` / `--memory` `-m` | 每 worker CPU / 内存限额 | `16` / `64Gi` |
| `--data` | 相对 `/workspace` 的数据路径 | — |
| `--shared-data` | 相对 `/datasets` 的共享只读数据集路径 | — |
| `--node` | 钉到指定节点(如 `wuji-3` 大内存机) | — |
| `--mount-path` | 团队 NAS 在容器里的挂载目录(不能是 `/`) | `/workspace` |
| `--shell` | 包裹命令的 shell(busybox/alpine 用 `sh`) | `bash` |
| `--cmd` | 整段命令字符串(与 `-- CMD` 二选一) | — |
| `--port` | 额外暴露的容器端口,可多次 | — |
| `--dry-run` | 只打印 Job YAML,不真提交 | 关 |

### SSH 参数

| 参数 | 说明 |
|---|---|
| `--ssh` | 在容器里起 sshd(root),之后 `wuji ssh` 进入 |
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
| `wuji exec <名>` | apiserver `exec` | 是 | 最省事,拿个交互 shell |
| `wuji ssh <名>`(默认) | `kubectl port-forward` 隧道 → `127.0.0.1` | 是 | VS Code Remote、`scp`/`rsync` |
| `wuji ssh <名>` + 提交时 `--expose-ssh` | NodePort → 节点弹性公网 IP,单跳直连 | **否** | 延迟/吞吐最好、大文件;只认 SSH 口令/密钥 |

> `--expose-ssh` 的 NodePort **绕过 apiserver 和 RBAC**,进去是 root 且能读写团队 NAS。安全组当前对
> `0.0.0.0/0` 开放 `30000-32767`;优先用密钥、用完 `wuji kill`。前提:目标节点已打
> `wuji.io/public-ip` 注解 + 安全组放行 NodePort 段。

## 保存 / 复用镜像

`wuji dev` 进去改完环境(装 apt 包、编 CUDA 组件等),一条 `wuji save` 把当前容器提交成镜像推
ACR,下次 `wuji dev/train -i` 直接用。你**不需要 docker,也不需要 ACR 凭据**——平台的
`wuji-saver` 组件在容器所在节点用 containerd 代做 commit + push,你只发一个保存请求。

```bash
wuji save mydev --tag myenv:v1     # 裸名自动补成 ACR 地址;也可给完整 registry 地址
wuji images                        # 看到 myenv:v1(kind=saved)
wuji train --name run1 -i wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com/wuji-rl/myenv:v1 -- python train.py
```

> 纯 Python 依赖建议装到 `/workspace/venv`(NAS,持久),那样根本不用存镜像。`wuji save` 主要用于
> **系统级**改动。

### `wuji save` 参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--tag` | 目标镜像。裸名 `myenv:v1` 自动补成 `<ACR>/wuji-rl/myenv:v1`;缺 `:tag` 补 `:latest` | 必填 |
| `--worker` | 第几个 worker(多机);pod `<名>-worker-<n>` | `0` |
| `--container` | 容器名(wuji 起的任务默认 `worker`) | `worker` |
| `--no-wait` | 只提交请求不等待完成(自己稍后查 ConfigMap 状态) | 关 |
| `--timeout` | 等 saver 完成的最长秒数 | `1800` |

目标 pod 必须存在且 `Running`,否则立即报错。安全模型与部署见
[`scheduler-platform/saver/README.md`](../saver/README.md)。

### `wuji images`

列出可直接拉取的镜像,读的是平台维护的镜像索引(共享 ConfigMap `wuji-public/wuji-images`),
**不需要 ACR 凭据**(ACR 拉取令牌本身没有列目录权限)。`--mine` 只看当前团队保存的,`-t <团队>`
看指定团队的(base 基础镜像始终显示)。ACR 完整权威目录仍走 `aliyun cr` / 控制台(管理员侧)。

## SDK 用法

```python
import wuji

wuji.submit(
    name="my-run", team="wuji-rl", queue="shared",
    image="wuji-rl-acr-registry.cn-huhehaote.cr.aliyuncs.com/wuji-rl/<你的镜像>:<tag>",
    gpus=8, nodes=1,
    command="python train.py",          # 或 ["python", "train.py"]
    data="mydata/ds",                    # → /workspace/mydata/ds, env WUJI_DATA
)

for j in wuji.list_jobs(team="wuji-rl"):
    print(j["name"], j["phase"], j["gpus"])

print(wuji.status("my-run", team="wuji-rl"))
print(wuji.logs("my-run", team="wuji-rl", tail=100))
wuji.kill("my-run", team="wuji-rl")

# 保存活容器为镜像(平台 wuji-saver 代做):
wuji.save_image("mydev", tag="myenv:v1", team="wuji-rl")   # 返回 {request, image, status, node, message}

# 列可拉取镜像索引:
for r in wuji.list_images(team="wuji-rl"):
    print(r["image"], r["kind"], r["team"])

# 只生成 manifest 不提交:
from wuji import build_volcano_job
manifest = build_volcano_job(name="x", team="wuji-rl", queue="shared",
                             image="<镜像>", gpus=8, command="python train.py")
```

## 备注

- 队列(如 `shared` / `pilot`)由管理员用 `scheduling.volcano.sh/v1beta1 Queue` 预建,提交时
  `--queue` 指定。
- `wuji data ls` 在集群外无法直接读 NFS,会起一个短命 helper Pod 挂对应 PVC 跑 `ls`,取日志后删掉。
- 没连集群 / 没 kubeconfig 时,命令给出清晰的中文报错而不是抛栈。
