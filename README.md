# PixelForge 节点包与共享镜像

本目录是节点包的**唯一仓库**：每个子目录一个节点包（`flowx.json` + `main.py` + `flowx_client.py`，
部分带 `server_op.py`/`ui/`/`mock.py`）。Studio 通过 `node import` 导入这些包；
docker 执行时从共享镜像 `lerkobba/flowx-pixelforge-nodes:v<BUNDLE_VERSION>` 的
`/opt/flowx-nodes/<name>/` 直接运行，不再从 Studio 拉取代码。


### B3（dev-plan §49）：SAM / GroundingDINO 分割 + 分块重绘超分

**权重（宿主 `D:\flowx-data\models`，容器内挂载 `/models`）**

| 目录 | 来源仓库 | 文件 | 体积 | 用途 |
| --- | --- | --- | --- | --- |
| `sam/vit_l` | `facebook/sam-vit-large` | `config.json` / `preprocessor_config.json` / `model.safetensors` | 1249MB | **默认** SAM 变体（精度/显存平衡，点选首次 1.65s） |
| `sam/vit_h` | `facebook/sam-vit-huge` | 同上 | 2564MB | 更准但慢（点选首次 6.17s，装载 163s） |
| `sam/vit_b` | `facebook/sam-vit-base` | 同上 | 375MB | 最省 |
| `grounding/tiny` | `IDEA-Research/grounding-dino-tiny` | `config.json` / `preprocessor_config.json` / `model.safetensors` / tokenizer 若干 | 690MB | **默认** 文本检测（0.6s/次） |
| `grounding/base` | `IDEA-Research/grounding-dino-base` | 同上 | 934MB | 更准 |

- 环境变量：`SAM_MODELS_DIR`（默认 `/models/sam`）、`GROUNDING_MODELS_DIR`（默认 `/models/grounding`）
- **零新增 Python 依赖**（`transformers==4.44.2` 已含 `SamModel`/`GroundingDinoForObjectDetection`，`timm` 由 B2 引入）⇒ **不重建推理镜像**
- 精度纪律：默认 **fp32**（Pascal GP104 非张量 fp16 吞吐 = fp32 的 1/64），变体参数可切 `fp16`/`auto`

**节点（7 个，id 212–218）**：`sam-load` / `sam-segment`（点·框选画布编辑器）/ `sam-auto-segment`（编号网格点选）
/ `grounding-dino-load` / `grounding-detect`（文本→框→SAM 像素级）/ `tiled-upscale-plan`（切片网格覆盖层）
/ `tiled-upscale-apply`（逐块重绘 + 权重归一化无缝融合 + 真实进度条）

**算子族**：`_common/server_ops/seg.py`（5 算子）、`_common/server_ops/tiled.py`（2 算子）；
模型经 `model_manager._register` 进 LRU（键 `sam:<variant>` / `gd:<variant>`，可 `model-unload`）。

**验收**：wf 133–138 / exec 649–666（local+docker 双路径）；`_tools/verify_seg.py`（15 项）与
`_tools/verify_tiled.py`（9 项）**FAILED = 0**；明细见 `flowx-dev-plan.md` §49.11。



### B4/S1（dev-plan §50）：DDColor 老照片上色

**权重（宿主 `D:\flowx-data\models\colorize\<variant>\`，容器内 `/models/colorize`）**

| 变体 | 上游仓库 | 体积 | 说明 |
| --- | --- | --- | --- |
| `modelscope` | `piddnad/ddcolor_modelscope` | 870MB | **默认**（清洗数据版，泛化最好） |
| `paper` | `piddnad/ddcolor_paper` | 870MB | 论文版 |
| `artistic` | `piddnad/ddcolor_artistic` | 870MB | 艺术风格版 |
| `paper_tiny` | `piddnad/ddcolor_paper_tiny` | 211MB | 最轻量（convnext-t） |

- 每个目录需 `config.json` + `pytorch_model.bin`；环境变量 `COLORIZE_MODELS_DIR`（默认 `/models/colorize`）
- **零新增依赖**（torch + timm + cv2 已有）⇒ **不重建推理镜像**；上游 5 个模块逐行内联进 `server_ops/colorize.py`
  的惰性工厂 `_ddclasses()`（保真对照：与官方 `ColorizationPipeline` **逐字节一致**）
- 节点：`colorize-load`（预热/换变体）/ `colorize`（上色，保留原图 L 通道只预测 ab，确定性可逐字节回归）
- 验收：`_tools/verify_colorize.py`（A1~A6，PASSED=26/FAILED=0）；明细见 dev-plan §50.8



### B4/S2（dev-plan §50）：电商模板与 `image-watermark`

- 新节点 **`image-watermark`**（纯节点侧 PIL，零 numpy）：文字/Logo 水印、九宫格定位、不透明度、描边、
  旋转、平铺（防盗图样张）、Logo 缩放；输出 `box(x,y,w,h)` 作为「非水印区逐字节不变」判据的锚点。
  ⚠️ 中文水印必须给 `font_path` 指向含 CJK 的 TTF（默认字体只有 ASCII）。
- **3 条模板流水线**（`pipeline/`）：`ecom-white-bg`（白底主图 1:1）、`ecom-scene`（场景换底）、
  `ecom-model-spec`（人脸修复 + RealESRGAN 4x + 水印）
- 复核脚本：`_tools/verify_ecom.py`（C1 白底纯度 / C2 尺寸规格 / C3 水印只改水印区 / C4 场景合成保主体）
- 运维事实：`matting-composite` 的 `feather=8` ≈ **24px 过渡带**（`feather` 在节点内部生效，
  落盘的 mask 是未羽化的）；复核要按"腐蚀后的核心"或"远离轮廓的背景"判定
- 明细见 dev-plan §50.9



### B4/S3（dev-plan §50.10–§50.11）：IC-Light 重打光

- **权重**（宿主 `/models`）：`iclight/iclight_sd15_fc.safetensors`（1639MB，默认）与
  `iclight/iclight_sd15_fbc.safetensors`（1639MB），来自 HF `lllyasviel/ic-light`（`gated=False`）；
  基座复用宿主既存的 **diffusers 目录格式 SD1.5** `/models/v1-5-pruned-emaonly-fp16`（零下载）。
- **算子族** `_common/server_ops/iclight.py`：`iclight.load`（变体→`iclight:<fc|fbc>`，进 LRU）
  + `iclight.relight`（重打光，两阶段）。关键机制（官方逐行等价）：UNet `conv_in` 扩到 8/12 通道、
  权重按 **delta（origin+offset）** `strict=True` 加载、猴子补丁 `unet.forward` 注入 `concat_conds`。
- **节点**：`iclight-load`（id 227）/ `iclight`（id 228）；`light_dir=none|left|right|top|bottom`，
  给了 `mask` 就按官方公式合成 127 灰底；`fbc` 可给 `bg` 图作为背景条件。
- **流水线**：`iclight-e2e-local`（wf 150）/ `iclight-e2e`（wf 151，docker 路径）。
- **复核**：`_tools/verify_iclight.py`（A1 尺寸 / A2 逐字节确定性 / A3 光方向物理判据 /
  A4 相对输入位移 / A5 再抠图 IoU / A6 highres 生效 / A7 执行器无关）——**17/17**。
- ⚠️ 常驻 4.07GB，与 flux（pinned 6.47GB）在 8GB 卡上互斥；fc/fbc 不要同时装载。
- ⚠️ 复现纪律：本族**显式传 `torch.Generator`** ⇒ 同种子逐字节可复现（含跨执行器）。



### B5/51.1：gate N —— `main()` 桩冒烟

`_tools/test-main-smoke.py`：对每个节点把 `flowx_client`/`image_io`/`executor_base` 换成桩，
用 flowx.json 参数默认值真跑一遍 `main()`。**为什么需要**：`node mock` 跑的是 `mock.py`，
跑不到 `main.py`（B4/S2 的 `NameError`、B4/S3 的可选参数问题都只在真机暴露）。
已挂进 `check-bundle.py`；负例自检 `--selftest` 5/5。当前 99 节点 = PASS 98 / SKIP 1 / FAIL 0。
改完 `main.py` 先跑：`python3 nodes/_tools/test-main-smoke.py <节点名>`。



### B5/51.3：InstantID 量化基线（`pipeline/instantid-ab-baseline.yaml`，wf 152）

受控 A/B（同底模/同参考脸/同 seed/同提示词，只改身份强度 0.8 → 0.0），用 `face.similarity`
度量身份余弦：**开启 0.7457 / 关闭基线 0.0356（Δ=+0.7101）** ⇒ 现有 InstantID 身份保持很强。
⚠️ 隧道下冷加载 SDXL(6.9GB)+CN(2.5GB) 实测 42 分钟 + >30 分钟；模型热了之后 512²/12 步
每分支约 6 分钟。⚠️ 失败执行会留 lease 钉住模型（`/model/unload` 的字段是 `target`，
对 pinned 项返回 `unloaded:(none)`），等 30 分钟自动过期。详见 dev-plan §51.3。



### B5/52：服务端插件归属簿修复 + G11 批量节点

- **服务端（已入库，未强制重启）**：修 `plugin_ops` 归属簿三个可证缺陷 —— ① 幽灵归属
  （手工删插件文件后归属簿仍指向它 ⇒ 假 409 且 DELETE 反 404）② 陈旧算子泄漏（同文件重传后
  改名/删掉的算子永久留在注册表）③ 回滚不补 `sys.modules`。新增
  `plugins.stale_owners` / `plugins.reconcile` / `plugins.plugin_module_name`，测试
  `inference-server/tests/test_plugin_registry.py`（15/15）。下次自然重启生效。
- **新节点（local-only，不进镜像）**：`image-batch-load`(229) / `image-batch-save`(230) /
  `image-montage`(231)；共享件 `_common/image_batch.py`、`_common/image_montage.py`。
- **流水线**：`batch-montage-preview`(wf 153) / `batch-roundtrip`(wf 154)。
- **复核**：`_tools/verify_batch.py`（10/10）：批量另存命名/等价性、拼图尺寸公式、格内逐字节保真。
- ⚠️ 服务端上传会**归一化图像编码**（灰度模式图会上传后重编码，像素不变）⇒ 判据写"等价"而非
  "字节一致"。⚠️ `image-montage` 的 `label=name` 必须接 `image-batch-load` 的 `names`。
- 明细见 dev-plan §52.1/§52.3/§52.4。



### B5/52.5：运维工具

- `_tools/purge-plugin-versions.py`（既有）：清理 `plugins.d` 里被取代的族版本文件（dry-run 默认）。
  本轮实测：插件文件 47→44、算子 113→104（删 seg 两个旧版 + tiled 一个旧版；当前版本完好）。
  未版本化的历史插件**不动**（老 bundle 还在用）。
- `_tools/index-e2e-out.py`（新增，**只读不搬文件**）：给 `~/flowx-e2e-out/` 建 `INDEX.md` ——
  按前缀报份数/最新一份/sha256/「不同 sha 个数」（= 同参数多次运行是否逐字节稳定的直接观测）。
  实测 143 组 / 269 文件 / 58.5MB。⚠️ 复核脚本按"最新一份"取产物 ⇒ 该目录**不能搬移归档**。


### B7/53：视频到视频（P2，v2v 换人/转绘）

- **节点（5 个）**：`load-video`(local-only，读本机原片并上传) / `video-to-frames` / `frames-to-video`
  / `video-repaint`（逐帧重绘，换人主力）/ `video-v2v`（整段 v2v）。
  算子族：`_common/server_ops/video_flow.py`（5 算子同文件，`OP_FAMILY=video_flow`）。
- **服务端改动**：`POST /videos`（原片字节入站 + 返回服务端路径）、
  `object_store._adopt_video_file`（VIDEO 预编码直通：拷进对象目录、不重编码）、
  `load_motion` 组合管 VAE dtype 二次断言（见下"真机定位缺陷"）。
  ⚠️ 这三处是**服务端代码**（非插件）⇒ 必须**重建/重启推理服务**才生效；
  算子本身走 `/admin/plugins` 热加载，无需重启。
- **帧序列契约**：帧批是**单个 IMAGE 对象的列表形态**（`GET /images/{id}?index=n` 取第 n 帧），
  节点间只传一个对象 ID（不是几十个 ID 拼串）。
- **真机定位缺陷（本批次修）**：`load_motion` 组合管出现「`vae.decode` 已被 `_decode_fp32`
  包装、VAE 权重却是 fp16」的不一致态 ⇒ `decode_latents` 里 conv2d 报
  `Input type (float) and bias type (c10::Half)`；**既有 t2v（`video.sample`）同样复现**。
  处置：① 核心 `load_motion` 在 offload 后二次断言 fp32；② 算子层 `align_vae_dtype()`
  （按 `inspect.ismethod` 判定"解码入口被包装过"，不按名字，抗多层调试包装）+
  `_VAEDtypeShim`（`force_upcast` 对外恒 False + encode/decode 出入参按真实 VAE 精度转换，
  拦住管道"按计算精度把 VAE 转 fp16"那一步）。
- **流水线**：`video-v2v-repaint.yaml`（跳舞视频换人：源片 + IPAdapter 参考人物 → 逐帧重绘）、
  `video-v2v-animatediff.yaml`（整段 v2v）。Studio wf 173/174。
- **验收**：`inference-server/tests/test_video_flow.py`（算子契约 + 抽帧/缩放纯逻辑 + VIDEO 直通，
  **本机可跑、零依赖**）；真机实测见 `inference-server/ROADMAP.md` 阶段 9.5。
- ⚠️ 逐帧重绘耗时 = 帧数 × 单帧耗时：先用 `max_frames=8` + `width=512` 小样验证效果再放开全片。
- ⚠️ **有效去噪步数（真机事故 2026-10-14）**：v2v 管道的 `strength` 是**时间轴截断比例**，
  有效步数 = `int(steps × strength)` —— `steps=3, strength=0.6` 实际只跑 **1 步**（DDIM linspace
  3 步 = [999,666,333]，截断后只剩 t=333），提示词完全无效、测出 ρ(源,出)=0.80~0.86（"没有油画风"）。
  处置：算子打印有效步数（=0 报错 / =1 告警）、`video-v2v` 新增 `enforce_inference_steps`
  （ComfyUI 语义）、模板配比改 25×0.7；逐帧重绘同口径（`resolve_span`，≈`int(steps×strength)`）。


## 单一事实源与三件套

| 文件 | 角色 |
| --- | --- |
| `Dockerfile` | **唯一事实源**：`COPY <dir>/*.py` 的行=该节点代码真的在镜像里 |
| `BUNDLE_VERSION` | 镜像 tag（构建与同步都读它） |
| `sync-flowx-json.py` | 按清单把 `image` / `executor.bundled` 写进各 `flowx.json`；清单外或 local-only 的节点清掉这两个字段 |
| `check-bundle.py` | **发布门槛**：反向校验声明与清单一致（声明 docker ⇒ 必须在清单里且有 image；写了 image/bundled ⇒ 必须在清单里；local-only ⇒ 不许声明 docker/image/bundled；声明上传控件 ⇒ 必须 local-only），并串跑 gate J/L 与共享件漂移校验 |
| `_tools/check-ops.py` | **发布门槛（gate J）**：算子族文件 `registry.register("<op>")` ↔ 成员节点 `main.py` 跑的算子一一对应、无孤儿、无跨文件重名；**F1 版本化**：族版本戳 == 内容 hash、`OP_FAMILY` == 文件 stem、`register()` 内注册必须在 `_VersionedRegistry` 包装之后 |
| `test-op-versioning.py` | F1 单测（假服务端 + 真族文件 exec）：新旧版本并存、旧版顶不掉新版、幂等、旧族兼容、基名自动翻译 |
| `_tools/purge-plugin-versions.py` | 清掉服务端 `plugins.d` 里**被取代**的族版本文件（默认 dry-run；判定依据是本地族版本戳，不碰历史未版本化文件） |
| `_tools/check-node-params.py` | **发布门槛（gate M，由 check-bundle.py 串跑）**：`main.py`/`mock.py` 里的 `param("<name>")`、`env` 的 `{{ Param.<name> }}` 引用的名字必须在 `flowx.json` 的 `parameters` 里；声明了但没人读的参数给警告。**真机事故教训**：`matting-refine` 参数 `min_area` 改名 `despeckle` 时 flowx.json 与 widget 都改了、`main.py` 忘了改，本地单测与产物校验都发现不了，直到真机 e2e 报 `TypeError: unexpected keyword argument` 才暴露 |
| `_tools/test-widget-ui.mjs` | **发布门槛（gate L，由 check-bundle.py 串跑）**：用 DOM stub 在 node 里真跑一遍 `ui/node-widget.js`——上传控件写回参数、绑定/回放态只读、未声明 `kind:'file'` 就不出上传入口、放大弹层的 ✕/点空白/Esc 三条关闭途径 ；**自定义 widget 通用回归**：凡 `ui/widget-def.js` 声明 `fields` 的节点（现 81 个）都按声明验控件渲染（slider/check/select/curves/textarea/text）、滑杆改动写回参数，声明 `replay` 的还验「▶ 预览」重放请求**原样带 `outputs.__op_name`** 走节点级代理 `/op`（F1 版本化算子名，dev-plan §47.9 A7；380 断言）|
| `probe-image-nodes.py` | 镜像内启动探针：detached 起容器 + `docker logs` 逐个确认节点能跑起来（隧道上 attach 模式的输出会被截断，故不用 `docker run` 前台模式） |
| `build-image.sh` | 构建 + 推送镜像（`SKIP_PUSH=1` 可只构建，指向远端 daemon 时用） |

漂移防线：**改了节点代码/清单后必须 `sync-flowx-json.py` → `check-bundle.py` 通过**，
再 bump 版本、构建镜像、重导节点包。历史上正是缺少这一步，导致 `image-upscale`/`detail-refine`
声明了 docker 但镜像里没有目录（执行时 `cd: /opt/flowx-nodes/xxx: No such file`）。

## 执行器分类（谁能进镜像/谁能跑 docker）

| 类 | 节点 | 声明 | 说明 |
| --- | --- | --- | --- |
| A 推理服务客户端 | 79 个（`checkpoint-loader`、`vae-load`、`controlnet-*`、`preprocess-*`（10 个，含 G7 预处理器族）、`image-upscale`、`detail-refine` …） | `supportedTypes: [local, docker]`，`preferredType: docker`，`bundled: true` + `image` | 只经 HTTP 调推理服务，镜像内 `python main.py` 即可跑（纯 stdlib） |
| B Studio 侧图像/蒙版/滤镜（PIL） | `image-rotate/flip/crop/composite`、`mask-*`（9 个）、G5 传统滤镜 8 个（`image-adjust` / `image-hsl` / `image-curves` / `image-blur` / `image-sharpen` / `image-noise` / `image-channel-extract` / `image-channel-merge`）、`image-pad`（扩画布） | 同上 | **图片本体经服务对象仓库传递**（`get_bytes /images/{id}` → PIL 计算 → `post_bytes /images`），不碰本地文件 ⇒ 进镜像 + 装 Pillow（离线 wheel，见下）；G5 滤镜族共享实现为 `_common/image_filters.py`（由 `_tools/sync-common.py` 分发，副本漂移即门槛失败） |
| C 读写宿主文件 | `save-image`、`save-video`、`load-image`、`load-video`、`mask-paint` | `supportedTypes: [local]`，无 `image`/`bundled` | 直接读写 Studio 宿主（手机）上的文件路径；容器内看不到 ⇒ **只能 local**。`load-image`/`mask-paint` 的本地路径字段声明了画布侧上传控件（widget `kind:'file'`），上传物落在 `~/flowx-input/uploads/` |
| — | `model-unload` | 常驻模型显式卸载（`[local, docker]`，偏好 docker） | 走 `POST /model/unload` 腾显存，配套 `inference-ensure` 的 `min_vram_mb` 闸门（dev-plan §21） |
| `_common`/`_tools`/`_widget-template` | — | 非节点目录，不进镜像 |

`sync-flowx-json.py` / `check-bundle.py` 里的 `LOCAL_ONLY` 常量就是 C 这份名单（`save-image`、`save-video`、`load-image`、`load-video`、`mask-paint`、`image-batch-load`、`image-batch-save`），改分类时两处一起改。

## docker 节点的服务地址必须用内网

容器内**访问不了公网隧道地址**（cpolar 不回环到自己的宿主机，实测 `5.tcp.cpolar.top:12073`
在容器里超时，`172.17.0.1:8100` 返回 200）。所以：

- workflow 里给 docker 节点的 `service_url` 绑 `{{ Param.service_url_internal }}`
  （值 `http://172.17.0.1:8100`，docker bridge 网关）；
- local 节点绑 `{{ Param.service_url }}`（公网隧道，因为 Studio 在手机上，只能走隧道）。

### 反向：画布侧读取要 `service_url_host`

有一类请求**不是节点自己发起的，而是 Studio 宿主机发起的**：模型名下拉（设计期代理
`/api/v1/service-proxy`）、实时预览帧（`/preview-frame` 中转拉帧）、▶预览重放/中断
（节点级代理 `/service-proxy`）。这些请求走的是 **Studio 的网络视角**，而 docker 节点的
`service_url` 是容器内网地址——Studio 根本连不上（症状：预览帧 502
`context deadline exceeded`、模型下拉 502）。

所以 `service_url_host` 是**必需的第三地址**（声明在 26 个有画布侧读取的节点里，
留空则回退 `service_url`）：

- docker 节点：`service_url`（容器内）+ `service_url_host: "{{ Param.service_url }}"`
  （Studio 侧走隧道）
- local 节点：两者相同，不填即可
- 节点侧用 `flowx_client.host_base(url)` 构造预览帧 URL 与上报 base；widget 侧优先读
  `service_url_host`

`check-bundle.py` 的校验 I 守住这条：有画布侧读取的节点必须声明该参数，
且直接上报预览帧的节点必须经 `host_base()` 构造 URL。

### G7 预处理器权重清单（`/models/preprocessors/`，宿主 `D:\flowx-data\models\preprocessors`）

G7 族算子（`preprocess.anyline/lineart/softedge/scribble/mlsd/normal/shuffle`）与既有
`preprocess.openpose/depth` 一样，**权重全部本地加载（零网络访问）**，缺文件时报错直接点名缺哪个文件。
任一权重缺失只影响对应那个算子，不影响其他算子与既有流水线。

| 节点 | 权重目录 | 必需文件 | 上游仓库 | 大小 |
| --- | --- | --- | --- | --- |
| `preprocess-anyline` | `anyline/` | `MTEED.pth` | `TheMistoAI/MistoLine` → `Anyline/MTEED.pth` | 0.25MB |
| `preprocess-lineart` | `lineart/` | `sk_model.pth`、`sk_model2.pth` | `lllyasviel/Annotators` | 17MB×2 |
| ↑（variant=anime） | `lineart_anime/` | `netG.pth` | `lllyasviel/Annotators` | 218MB |
| ↑（variant=standard） | — | 无权重（纯算法，无需下载） | — | — |
| `preprocess-softedge`（pidi） | `pidi/` | `table5_pidinet.pth` | `lllyasviel/Annotators` | 2.9MB |
| ↑（variant=hed） | `hed/` | `ControlNetHED.pth` | `lllyasviel/Annotators` | 29MB |
| `preprocess-scribble` | `pidi/` | `table5_pidinet.pth`（与 softedge 共用） | `lllyasviel/Annotators` | — |
| `preprocess-mlsd` | `mlsd/` | `mlsd_large_512_fp32.pth` | `lllyasviel/Annotators` | 6.3MB |
| `preprocess-normal` | `normalbae/` | `scannet.pt` | `lllyasviel/Annotators` | 291MB |
| `preprocess-shuffle` | — | 无权重（纯算法） | — | — |

> G7 族算子实现集中在 `_common/server_ops/preprocess_g7.py`（7 算子一个文件，由 `_tools/sync-common.py` 分发到 7 个节点目录）；三算子族 canny/depth/openpose 仍在 `_common/server_ops/preprocess.py`——**两族必须分开**：服务端上传文件名 = 算子名（点换下划线），跨文件重名算子一律 409 全量回滚。
| （既有）`preprocess-openpose` | `openpose/` | `body_pose_model.pth`、`hand_pose_model.pth`、`facenet.pth` | `lllyasviel/Annotators` | 510MB |
| （既有）`preprocess-depth` | `midas/` | `dpt_hybrid-midas-501f0c75.pt` | `lllyasviel/Annotators` | 493MB |

下载与落盘（本机无 GPU、无外网的环境：手机侧下载后离线送入宿主机）：

```bash
# 1) 有外网的一侧下载（huggingface.co；国内可换 hf-mirror.com，但 resolve 会 302 回 HF，
#    实际要能连 huggingface.co）
curl -L -O https://huggingface.co/lllyasviel/Annotators/resolve/main/scannet.pt
# 2) 送入宿主机 D:\flowx-data\models\preprocessors/<子目录>/（大文件用分块或局域网 HTTP，
#    docker cp 对 >100MB 文件在此隧道上会卡死）
```

### B2 权重清单（BiRefNet 抠图 + 人脸修复）

两类权重都是**本地加载、零网络访问**，缺文件时报错点名缺哪个文件与目标目录。

| 目录（`/models/`，宿主 `D:\flowx-data\models`） | 必需文件 | 上游 | 大小 | env 覆盖 |
| --- | --- | --- | --- | --- |
| `matting/lite/` | `config.json`、`birefnet.py`、`BiRefNet_config.py`、`model.safetensors` | HF `ZhengPeng7/BiRefNet_lite` | 178MB | `MATTING_MODELS_DIR` |
| `matting/main/` | 同上 4 件 | HF `ZhengPeng7/BiRefNet` | 445MB | 同上 |
| `matting/portrait/` | 同上 4 件 | HF `ZhengPeng7/BiRefNet-portrait` | 885MB | 同上 |
| `facerestore/` | `GFPGANv1.3.pth`（或 v1.2）、`RestoreFormer.pth` | TencentARC GitHub release | 349MB / 291MB | `FACERESTORE_MODELS_DIR` |
| `insightface/` | antelopev2 模型包（`models/antelopev2/*.onnx`） | insightface 官方 zip | 352MB | `INSIGHTFACE_HOME` |

要点：

- **matting 的 4 个文件必须同目录**：后三个是 `trust_remote_code` 的自定义建模代码，
  `config.json` 里 `bb_pretrained=false` ⇒ 不会再拉 backbone 预训练权重（离线可用）。
  下载后可用 `python -c "import json,struct;..."` 校验 safetensors 头（见 `_e2e` 复核脚本）。
- **facerestore 只支持 GFPGAN / RestoreFormer**：spandrel 0.4.2（当前最新发布版）的
  wheel 里没有 CodeFormer 架构，`face.restore(model='codeformer')` 会明确报错（dev-plan §48.7）。
- insightface 模型包（antelopev2）：`FaceAnalysis` 首次调用会自动下载到 `INSIGHTFACE_HOME`；
  **容器必须挂载 `/models`**，否则会下到容器临时层、重建即丢（踩过）。

**TEED 未纳入**：`TheMistoAI/TEED` 仓库已私有（401），controlnet_aux 的 `TEEDdetector`
在 0.0.10 里仅暴露类、拿不到可分发权重；软边缘需求由 `softedge(pidi/hed)` 覆盖。

**seg 未纳入**：controlnet_aux 0.0.10 无语义分割检测器（`segment_anything` 的 SamDetector
属 G2 批次），seg 语义由 G2 批次的 SAM 节点承接（见 dev-plan §四十四 G2）。

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
python3 check-bundle.py          # 串跑 gate J/L/M（算子族一致性 / widget 交互 / 参数引用一致性）
#    只改 widget（_common/widget-base.js 或某个 ui/widget-def.js）时：
python3 _tools/build-widget.py --all   # 重建产物（不重建 gate G 会拦）
node _tools/test-widget-ui.mjs         # 交互回归（可选，check-bundle 已包含）
# 3. bump 版本
echo -n 1.6.3 > BUNDLE_VERSION && python3 sync-flowx-json.py && python3 check-bundle.py
# 4. 构建（Docker Desktop 在本机时直接跑；远端 daemon 用 DOCKER_HOST 指向它）
DOCKER_HOST=tcp://<daemon>:2375 SKIP_PUSH=1 ./build-image.sh
# 5. 镜像内启动探针（可选但推荐）
DOCKER_HOST=tcp://<daemon>:2375 python3 probe-image-nodes.py
# 6. Studio 侧重导改动的节点包（同名同版本原地更新，流水线里的 name@version 引用无需改）
flowx-studio node import --type folder --path nodes/<name> --overwrite
```

## 画布侧 widget（`ui/node-widget.js`）

每个节点包的画布控件是**预编译单文件 bundle**，由 `_common/widget-base.js`（通用渲染器）
`+ ui/widget-def.js`（本节点声明）+ 页脚拼接而成（`_tools/build-widget.py`）。

- **能力共享、展示声明制**：控件形态由 `widget-def.js` 的 `kind` 声明，渲染器只提供实现。
  现支持 `text` / `slider` / `select` / `model`（模型下拉，需 `service_url_host`）/ `curves`
  / `paint`（蒙版手绘）/ `file`（本机文件上传）；**未声明的节点界面上不会出现该入口**
  （不做“看到 `*_path` 就自动加上传按钮”这类猜测——`save-image.output_dir` 会被误伤）。
- `kind:'file'` 落地上传：跟 Studio 同源 `POST /api/v1/media/upload`（multipart、流式落盘、
  扩展名白名单、图片 32MB/视频 512MB），返回本机路径写回参数。**因此只能用在 local-only 节点**
  （gate K）：路径在 FlowX 那台机器上，docker/远端执行器看不见。上传目录 = Studio `media.roots`
  里名为 `flowx-input` 的根下面的 `uploads/`（默认 `~/flowx-input/uploads/`）。
- 预览/结果区直读本机文件（同源媒体端点 `/api/v1/media/file`），点击进可缩放弹层；
  弹层生命周期集中在 `mount` 顶部共用（`setLightbox/closeLightbox`），**三条关闭途径
  （✕ / 点遮罩空白 / Esc）必须全部可用**——1.3.0 因为漏传 `onClose` 且 Esc 监听挂在
  `mediaImageParam` 分支里，出过“预览图放大后关不掉”的事故，现由 gate L 守住。
- 改了 `widget-base.js` 或任一 `widget-def.js` ⇒ **必须重建全部 bundle**（`--all`），
  否则 gate G 拦；重建后重导受影响节点（`?v=<updatedAt>` 变化，画布刷新即取新 bundle）。

```bash
python3 _tools/build-widget.py --all --check   # 只校验产物是否最新
node _tools/test-widget-ui.mjs                  # 交互回归（自动扫描声明 kind:'file'/media*Output 的节点）
```

## 已知的长期抖动（未修，记录在案）

`flowx_client.py` 由各节点目录各持一份副本，目前有 3 个变体（按功能子集裁剪：
带/不带 `ensure_plugin`、`emit_preview` 是否带 `base`/`job_id`）。调用 `ensure_plugin`
的节点都自带该函数，不影响运行；但若后续要统一升级客户端（如新增协议字段），
需要按目录逐个同步，或改为构建期注入单一副本。

## 算子族文件与 F1 版本化（算子身份 = 实现内容，dev-plan §47）

族文件（`_common/server_ops/<族>.py`，由 `_tools/sync-common.py` 分发为成员节点的
`server_op.py`）顶部有 F1 合同，**改族文件时必须遵守**：

```python
OP_VERSION = "42636b08e915"   # 由 sync-common.py 依本文件内容自动改写（幂等），禁止手改
OP_FAMILY = "preprocess_g7"   # 必须等于文件名 stem（gate J 校验）

def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)   # ← 必须；注册名变 <基名>@<版本>
```

- **版本 = 族文件内容 hash 前 12 位**（计算时把 stamp 行归零，避免自引用）。改一行实现 ⇒
  换版本 ⇒ 换算子名（`preprocess.anyline@42636b08e915`）⇒ 换落盘文件
  （`preprocess_g7_42636b08e915.py`）。**没改的族跨版本同名**，不产生新条目。
- **落盘名内容寻址**：同内容 = 同文件 = 幂等（并发自注册不再撞 409）；不同内容 = 不同文件 =
  **新旧版本在服务端并存、互不顶替**（旧版跑多少次也顶不掉新版）。
- **节点侧不用写版本**：`ensure_plugin` 返回服务端确认的实际算子名，`flowx_client` 内部把
  基名翻译成实际名（`call_op`/`submit_job` 自动翻译）——节点 `main.py` 照旧写
  `run_op("preprocess.anyline", …)` / `call_op(url, "sd.sample", …)` 即可。
- **老 bundle 兼容**：未版本化的历史算子（如 `sd.sample`、`preprocess.canny` 所在的旧插件文件）
  继续按基名工作；两种命名并存，无需停机或强制重导。
- **清理**：`plugins.d` 会随发版累积版本文件，用
  `python3 _tools/purge-plugin-versions.py [--apply]` 只删被取代的旧版本（默认 dry-run）。
- **例外**：`inference-op`（手工逃生舱）不走 `ensure_plugin`，其 `op_name` 参数要填
  `/ops` 里的**实际名**（含版本后缀）。

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

## "改完必跑"清单（2026-10-13 起，dev-log §58~§60 教训固化）

改**任何**节点/共享件后，按顺序跑（全部退出码 0 才算完）：

```bash
python3 nodes/_tools/sync-common.py            # 1) 分发共享件（改过 _common/ 才需要）
python3 nodes/check-bundle.py                  # 2) 发布门槛：清单 A~D + 共享件 F + gate
                                               #    G/H/I/J/K/L/M/N/O/P（含 flowx.json 结构、
                                               #    算子契约、widget 交互、参数一致）
python3 nodes/_tools/test-main-smoke.py        # 3) 每个节点真跑一遍 main()（桩）
node nodes/_tools/test-widget-ui.mjs           # 4) 画布控件交互回归（DOM stub）
python3 nodes/_tools/reimport-nodes.py --check # 5) Studio 快照巡检（版本/镜像 tag/执行器）
```

**提交纪律（同仓库并行开发时）**：提交用**显式路径**（`git add <file> …`），不要 `git add -A`
—— 仓库里可能同时有别人的在制品（本轮实测：并行开发的视频节点 + inference-server 改动）。

**超时三层（本轮两次真机事故的总结）**：`节点 timeout ≥ job_timeout ≥ 排队+计算`。
- 排队预算：任务超时必须盖住"排在数小时长任务之后"的等待（§58.5/§60.11）；
- 节点超时必须盖住任务超时（§61.6：缺省走 Studio 默认 ~1h，把 82 分钟的节点杀成 failed，
  而它的 job 是 done）。gate P 现在硬校验这两条。

**顺序纪律（镜像 tag 写在每个 flowx.json 里，这条最容易踩）**：

```
改节点 → bump nodes/BUNDLE_VERSION → sync-flowx-json.py → build-image.sh
       → reimport-nodes.py（全量！99 份 flowx.json 都变了）
```

漏掉最后一步 ⇒ docker 执行器按快照里的旧 tag 拉**旧镜像**（实测 exec 721/728 都栽在这，
traceback 直接指出来）。另外三条本轮实测的坑：
- **对象仓库有 TTL**：临时工作流别硬编码历史对象 id（现取句柄或用 `load-image` 现传）；
- **Studio 限流 100 req/min**：重导/翻页核对都要节流（`reimport-nodes.py` 已内置 1.25s，核对要退避）；
- **模型租约**：失败/中断的执会留 30 分钟租约且其常驻条目不被驱逐 ⇒ 后续冷加载等 300s 后失败。
  排查入口 `vram-ops`（`action=leases`）或 `GET /leases`；清理 `action=unload`（可带 `lease`）。

## 新增运维/门禁工具（本轮）

| 工具 | 用途 |
| --- | --- |
| `_tools/stub-inference.py` | 推理服务**离线桩**（`/ops` 视为内置、`/jobs` 可配 pending、记账同步 `/op` 次数与 `/interrupt`）。"节点改协议"类改动的确定性门禁（先例 §51.3 / §58.3） |
| `_tools/reimport-nodes.py [--check]` | 节流重导（48/分钟）/ 只巡检快照漂移 |
| `_tools/gen-instantid-sweep.py` | 生成 InstantID 受控扫参工作流（weight × cn_strength 网格 + 逐点 face-similarity 打分） |
| `nodes/vram-ops/` | 显存/租约/策略运维节点（status/leases/unload/interrupt/policy_get/policy_set） |
| `nodes/vram-ops/test_main.py` | 节点自带 HTTP 桩测试（断言请求形状/头/输出，20 项） |
| `inference-server/app/policy.py` + `tests/test_policy.py` | 显存策略旋钮（不 import torch，本机可单测；端点**需重新部署服务**才生效） |

## 取消语义（dev-log §60.2，两条路径 + 幂等）

节点被取消时，**两条路同时**保证远程任务不孤儿，实测两次都命中同一 job 属正常（`/interrupt`
按 job_id 幂等）：

1. **节点自清**：`flowx_client._register_inflight` 接管 SIGTERM/SIGHUP ⇒ `POST /interrupt`；
2. **Studio 兜底**：`interruptRemoteJobs` 按节点上报的 **job 标记**（`FLOWX_PREVIEW` 的
   `base`+`job_id`）转发中断 —— 所以**凡是提交 job 的节点都必须上报标记**（7 个 loader/vae
   节点本轮补齐；否则节点信号一旦输给执行器杀进程，任务就变孤儿）。

注意：中止一个**正在跑**的远程算子，服务端在**下一个检查点**（通常是步边界）才响应
`check_cancelled()`——中途看到的 `running` 是设计，不是失败。
