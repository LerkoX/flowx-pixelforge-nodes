# PixelForge 节点包与共享镜像

本目录是节点包的**唯一仓库**：每个子目录一个节点包（`flowx.json` + `main.py` + `flowx_client.py`，
部分带 `server_op.py`/`ui/`/`mock.py`）。Studio 通过 `node import` 导入这些包；
docker 执行时从共享镜像 `lerkobba/flowx-pixelforge-nodes:v<BUNDLE_VERSION>` 的
`/opt/flowx-nodes/<name>/` 直接运行，不再从 Studio 拉取代码。

## 单一事实源与三件套

| 文件 | 角色 |
| --- | --- |
| `Dockerfile` | **唯一事实源**：`COPY <dir>/*.py` 的行=该节点代码真的在镜像里 |
| `BUNDLE_VERSION` | 镜像 tag（构建与同步都读它） |
| `sync-flowx-json.py` | 按清单把 `image` / `executor.bundled` 写进各 `flowx.json`；清单外或 local-only 的节点清掉这两个字段 |
| `check-bundle.py` | **发布门槛**：反向校验声明与清单一致（声明 docker ⇒ 必须在清单里且有 image；写了 image/bundled ⇒ 必须在清单里；local-only ⇒ 不许声明 docker/image/bundled） |
| `probe-image-nodes.py` | 镜像内启动探针：detached 起容器 + `docker logs` 逐个确认节点能跑起来（隧道上 attach 模式的输出会被截断，故不用 `docker run` 前台模式） |
| `build-image.sh` | 构建 + 推送镜像（`SKIP_PUSH=1` 可只构建，指向远端 daemon 时用） |

漂移防线：**改了节点代码/清单后必须 `sync-flowx-json.py` → `check-bundle.py` 通过**，
再 bump 版本、构建镜像、重导节点包。历史上正是缺少这一步，导致 `image-upscale`/`detail-refine`
声明了 docker 但镜像里没有目录（执行时 `cd: /opt/flowx-nodes/xxx: No such file`）。

## 执行器分类（谁能进镜像/谁能跑 docker）

| 类 | 节点 | 声明 | 说明 |
| --- | --- | --- | --- |
| A 推理服务客户端 | 37 个（`checkpoint-loader`、`vae-load`、`controlnet-*`、`preprocess-*`、`image-upscale`、`detail-refine` …） | `supportedTypes: [local, docker]`，`preferredType: docker`，`bundled: true` + `image` | 只经 HTTP 调推理服务，镜像内 `python main.py` 即可跑（纯 stdlib） |
| B Studio 侧图像/蒙版（PIL） | `image-rotate/flip/crop/composite`、`mask-*`（9 个） | 同上 | **图片本体经服务对象仓库传递**（`get_bytes /images/{id}` → PIL 计算 → `post_bytes /images`），不碰本地文件 ⇒ 进镜像 + 装 Pillow（离线 wheel，见下） |
| C 读写宿主文件 | `save-image`、`save-video`、`load-image` | `supportedTypes: [local]`，无 `image`/`bundled` | 直接读写 Studio 宿主（手机）上的文件路径；容器内看不到 ⇒ **只能 local** |
| — | `model-unload` | 常驻模型显式卸载（`[local, docker]`，偏好 docker） | 走 `POST /model/unload` 腾显存，配套 `inference-ensure` 的 `min_vram_mb` 闸门（dev-plan §21） |
| `_common`/`_tools`/`_widget-template` | — | 非节点目录，不进镜像 |

`sync-flowx-json.py` / `check-bundle.py` 里的 `LOCAL_ONLY` 常量就是 C 这份名单（`save-image`、`save-video`、`load-image`），改分类时两处一起改。

## docker 节点的服务地址必须用内网

容器内**访问不了公网隧道地址**（cpolar 不回环到自己的宿主机，实测 `5.tcp.cpolar.top:12073`
在容器里超时，`172.17.0.1:8100` 返回 200）。所以：

- workflow 里给 docker 节点的 `service_url` 绑 `{{ Param.service_url_internal }}`
  （值 `http://172.17.0.1:8100`，docker bridge 网关）；
- local 节点绑 `{{ Param.service_url }}`（公网隧道，因为 Studio 在手机上，只能走隧道）。

## Pillow：离线 wheel

远端 docker 宿主**没有外网**（`pip install` 报 `Network is unreachable`），故把
manylinux cp311 wheel 放在 `_wheels/` 随构建上下文带进去离线安装：

```bash
pip download Pillow==12.3.0 --only-binary=:all: --no-deps \
  --platform manylinux_2_28_x86_64 --python-version 3.11 --implementation cp --abi cp311 -d _wheels
```

升级 Pillow 时重跑上面这条并同步 Dockerfile 的版本号即可（只有这 9 个 PIL 节点用它）。

## 加/改一个节点包（发布流程）

```bash
cd nodes
# 1. 改节点包代码或 flowx.json（新节点记得加进 Dockerfile COPY 清单）
# 2. 同步 + 门槛校验（必须通过）
python3 sync-flowx-json.py
python3 check-bundle.py
# 3. bump 版本
echo -n 1.6.3 > BUNDLE_VERSION && python3 sync-flowx-json.py && python3 check-bundle.py
# 4. 构建（Docker Desktop 在本机时直接跑；远端 daemon 用 DOCKER_HOST 指向它）
DOCKER_HOST=tcp://<daemon>:2375 SKIP_PUSH=1 ./build-image.sh
# 5. 镜像内启动探针（可选但推荐）
DOCKER_HOST=tcp://<daemon>:2375 python3 probe-image-nodes.py
# 6. Studio 侧重导改动的节点包（同名同版本原地更新，流水线里的 name@version 引用无需改）
flowx-studio node import --type folder --path nodes/<name> --overwrite
```

## 已知的长期抖动（未修，记录在案）

`flowx_client.py` 由各节点目录各持一份副本，目前有 3 个变体（按功能子集裁剪：
带/不带 `ensure_plugin`、`emit_preview` 是否带 `base`/`job_id`）。调用 `ensure_plugin`
的节点都自带该函数，不影响运行；但若后续要统一升级客户端（如新增协议字段），
需要按目录逐个同步，或改为构建期注入单一副本。

## 共享件单一事实源（`_common/` → 各节点目录）

`flowx_client.py` 曾分裂成 **3 个变体**（有无 `ensure_plugin`、`emit_preview` 新旧签名），
改一次协议字段要逐目录改、极易漂移。现在：

- 唯一事实源 = `_common/flowx_client.py`（超集：含 `ensure_plugin` + 新版
  `emit_preview(url, progress, tok, base, job_id)`）与 `_common/executor_base.py`；
- 各节点目录里的副本由 `_tools/sync-common.py` 生成（`--check` 只校验不改）；
- `check-bundle.py` 把"副本 == _common"作为**发布门槛**（漂移即非 0 退出），
  与 Dockerfile 清单校验收在同一道关卡里。

```bash
python3 nodes/_tools/sync-common.py          # 分发共享件
python3 nodes/_tools/sync-common.py --check  # 只校验漂移
python3 nodes/check-bundle.py                # 清单 + 共享件双重校验（发布门槛）
```
