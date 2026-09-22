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
| A 推理服务客户端 | 37 个（`checkpoint-loader`、`vae-load`、`controlnet-*`、`preprocess-*`、`image-upscale`、`detail-refine` …） | `supportedTypes: [local, docker]`，`bundled: true` + `image` | 只经 HTTP 调推理服务，镜像内 `python main.py` 即可跑（无第三方 pip 依赖，纯 stdlib） |
| B 读写宿主文件 | `save-image`、`save-video`、`load-image` | `supportedTypes: [local]`，无 `image`/`bundled` | 直接读写 Studio 宿主（手机）上的文件路径；容器内看不到这些文件 ⇒ **只能 local** |
| C Studio 侧图像/蒙版 | `image-rotate/flip/crop/composite`、`mask-*`（9 个） | `supportedTypes: [local]` | 需要 Pillow，而镜像刻意不带 Pillow（省体积）；处理的是 Studio 本地图 ⇒ local-only |
| — | `_common`/`_tools`/`_widget-template` | — | 非节点目录，不进镜像 |

`sync-flowx-json.py` / `check-bundle.py` 里的 `LOCAL_ONLY` 常量就是 B+C 这份名单，改分类时两处一起改。

## 加/改一个节点包（发布流程）

```bash
cd nodes
# 1. 改节点包代码或 flowx.json（新节点记得加进 Dockerfile COPY 清单）
# 2. 同步 + 门槛校验（必须通过）
python3 sync-flowx-json.py
python3 check-bundle.py
# 3. bump 版本
echo -n 1.6.1 > BUNDLE_VERSION && python3 sync-flowx-json.py && python3 check-bundle.py
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
