# 商业许可（flowx-pixelforge）

FlowX 采用**双许可（Dual Licensing）**模式发布：

| 通道 | 许可证 | 价格 | 适用场景 |
|---|---|---|---|
| **社区版** | [GPL-3.0-or-later](./LICENSE) | 免费 | 开源项目、内部自用、可接受开源义务 |
| **商业版** | 商业许可协议（需单独签署） | 按需报价 | 无法接受 GPL 开源义务的商业场景 |

**同一份代码，两种协议。** 商业许可购买的不是额外的源码，而是**免除 GPL-3.0
开源义务的权利**，以及商业级的保障条款。

---

## 1. 我需要商业许可吗？

以下情形**必须**取得商业许可，否则将违反 GPL-3.0 的 copyleft 条款：

| 场景 | 说明 |
|---|---|
| **闭源集成** | 把 FlowX 嵌入、链接进你的闭源产品并对外分发 |
| **私有化交付** | 向客户/私有云交付 FlowX 或其修改版，但不希望交出全部对应源码 |
| **SaaS 私有部署** | 客户要求签署 NDA、源码不得扩散 |
| **专利与合规** | 需要明确的专利授权与知识产权赔偿（GPL-3.0 第 15–16 条明确免责） |
| **需要担保与 SLA** | GPL 不提供任何形式的担保或服务水平承诺 |
| **白标 / OEM 转售** | 以自己的品牌分发 FlowX 或其衍生作品 |

**以下情形无需商业许可，可继续免费使用社区版：**

- 在 GPL-3.0 兼容的开源项目中集成
- 公司内部自用（不改动对外分发）
- 自建 SaaS 服务端（GPL-3.0 不含网络服务条款；AGPL 才有）
- 学习、研究、评估

> ⚠️ 若你的使用方式会触发 GPL 的开源义务，而你又不打算开源自己的代码，
> 那么**唯一合规的路径**是取得商业许可。请不要在未获许可的情况下闭源分发。

## 2. 商业许可包含什么

- 免除 GPL-3.0 的开源与源码提供义务，可用于闭源产品
- 允许在专有条款下修改、集成、分发、再许可（`sublicense`）
- 明确的**专利授权**（覆盖授权方拥有且被本软件必然侵犯的专利权利要求）
- **知识产权赔偿（IP Indemnification）**：第三方主张本软件侵犯其著作权/专利时
  由授权方承担抗辩与赔偿责任（具体范围以合同为准）
- 约定的**技术支持与 SLA**
- 可选：优先修复、定制开发、架构咨询

## 3. 商业许可**不**包含什么

为避免误解，以下内容**不在**商业许可范围内，需另行处理：

| 项目 | 说明 |
|---|---|
| **商标权** | 商业许可不授予使用 `FlowX` 名称、Logo 的权利，需单独书面约定 |
| **第三方组件** | 本项目依赖的第三方开源组件（如 Diffusers / Apache-2.0、Go 生态库）仍受其各自许可证约束 |
| **ComfyUI 及其生态** | ComfyUI 由 Comfy Org 以 GPL-3.0 发布，**本项目无权对其再授权**。若需闭源使用 ComfyUI 本身，须向 [Comfy Org](https://www.comfy.org/) 取得授权。本项目仅通过**进程边界**（HTTP / 子进程）与其交互 |
| **模型权重** | 各类预训练模型的权重（如 FLUX.1-dev 非商业许可、SDXL CreativeML OpenRAIL++-M 等）受其**自身许可证**约束，与本项目许可证无关。商业使用前请逐一核实权重许可证 |
| **无担保的免责** | 商业许可提供约定范围内的赔偿，但不对「适配特定业务目的」作绝对保证 |

## 4. 定价模式

商业许可按以下任一方式计费（具体以报价单为准）：

- **按开发者席位**：适合内部研发团队
- **按部署实例**：适合私有化交付、OEM
- **年度订阅**：含技术支持与升级
- **营收分成**：适合以 FlowX 为核心产品的深度合作

初创公司、开源项目、教育与非营利机构可申请优惠或免费商业许可，欢迎来函说明。

## 5. 如何获取

请发送邮件至 **chenyingqiao19931115@gmail.com**，注明：

1. 公司/组织名称与所在地
2. 计划的使用方式（闭源集成 / 私有化交付 / OEM 等）
3. 预计部署规模（席位数 / 实例数）
4. 是否需要技术支持与 SLA

我们会在 3 个工作日内回复报价与许可协议草案。

## 6. 关于 GPL 版本的持续有效

**已经以 GPL-3.0-or-later 发布的版本，其开源授权不可撤销。** 任何人都可以继续
按 GPL-3.0 使用、修改、分发这些版本（只要其自身合规）。

双许可的含义是：**未来的版本**会继续同时提供 GPL 与商业两条通道，而不是
「先开源后转闭源」。本项目不会撤回已发布的开源授权。

## 7. 贡献者

商业许可的可持续性依赖于对全部代码的再授权能力。因此**任何贡献都需要先签署
[贡献者许可协议（CLA）](./CLA.md)**：

- 你**不会**转让著作权，仍然拥有你的代码
- 你授予项目方将你的贡献以任意许可证条款（含专有商业许可）再许可与分发的权利

详见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

---

## English Summary

FlowX is **dual-licensed**: the community edition is available under
**GPL-3.0-or-later** for free, and a separate **commercial license** is available
for organizations that cannot comply with GPL's copyleft obligations — such as
embedding FlowX in a proprietary product, on-premise delivery under NDA, OEM
rebranding, or requiring patent grants and IP indemnification.

A commercial license does **not** grant trademark rights, does **not** cover
third-party components or model weights (each governed by its own license), and
does **not** extend to ComfyUI itself, which is licensed by Comfy Org under
GPL-3.0 and cannot be relicensed by this project.

GPL grants already made are irrevocable: previously released versions remain
free to use under GPL-3.0. Dual licensing applies to future releases — this
project will not withdraw open source licensing from code already published.

Contact **chenyingqiao19931115@gmail.com** for pricing and terms.
