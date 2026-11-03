# 贡献指南

感谢你有兴趣为 FlowX 做贡献！在开始之前，请先花两分钟读完本文档——尤其是
第 1 节，它是**提交 Pull Request 的前置条件**。

---

## 1. 提交 PR 前必须签署 CLA

本项目采用 **「开源 + 商业」双许可**模式：

| 通道 | 许可证 | 面向 |
|---|---|---|
| 社区 | GPL-3.0-or-later | 所有人，免费 |
| 商业 | 单独的商业许可协议 | 无法接受 GPL 开源义务的客户 |

为了让这套模式能够持续运转，项目方必须**拥有对全部代码的再授权能力**。
因此，任何贡献都需要先签署 **[贡献者许可协议（CLA）](./CLA.md)**。

**请特别注意 CLA 的核心条款**（第 2 条）：你**不会**转让著作权，你仍然拥有
你的代码；但你授予项目方将你的贡献以**任意许可证条款**（含专有商业许可证）
再许可与分发的权利。如果你不接受这一点，请不要提交贡献——你可以自由地
fork 本项目并按 GPL-3.0 自行使用和分发。

### 签署方式

首次提交 Pull Request 时，CLA Assistant 机器人会自动在 PR 中留言提示。
在 PR 评论区回复以下内容即可完成电子签署：

```
I have read the CLA Document and I hereby sign the CLA
```

> ⚠️ **职务作品提醒**：如果你的贡献是在受雇期间、为雇主业务之目的作出的，
> 或依法律构成职务作品，你**必须先获得雇主的书面授权**，或由雇主单独签署
> `CLA-Corporate.md`。请勿在未获授权的情况下提交。

---

## 2. 贡献流程

1. **先开 Issue 讨论**。较大的改动（新增节点、接口变更、重构）请先开 issue
   说明动机与方案，避免你写完 PR 后才发现方向不一致。
2. **Fork 并创建特性分支**。分支名建议使用 `feat/xxx`、`fix/xxx`、`doc/xxx`。
3. **一个 PR 只做一件事**。不要把无关的重构、格式化、功能变更混在同一个 PR 里。
4. **确保构建与测试通过**。
   ```bash
   pytest -q                 # 单元测试
   # e2e 用例位于 tests/ 下的 YAML，按需在 FlowX Studio 中运行
   ```
5. **提交 PR**，清晰描述：改了什么、为什么改、如何验证。

## 3. 提交信息规范

本项目使用 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<可选作用域>): <简述>

<可选的正文：说明动机与实现要点>

<可选的脚注>
```

常用 `type`：`feat` / `fix` / `docs` / `refactor` / `test` / `chore` / `perf`。

示例：

```
fix(docker): peek attach 流首帧识别 'already running' 哨兵

daemon 对并发 attach 的拒绝不写 HTTP 错误码，而是塞进劫持流内容，
peek 1.5s 窗口检出哨兵则关闭重试。
```

## 4. 代码规范

- **Python 代码遵循 PEP 8**，使用 4 空格缩进
- 节点需提供 `flowx.json` 元数据，声明输入/输出与依赖
- 新增功能请补充对应测试；测试应可重复运行、不依赖外部网络
- **单个文件建议不超过 1000 行**，超长文件请考虑拆分
- 不要在代码、注释、测试数据中硬编码密钥、Token、内网地址，也不要硬编码模型绝对路径

## 5. 文档

- 面向用户的行为变更，请同步更新 `README.md`
- 节点清单与元数据变更请同步更新 `nodes/BUNDLE_VERSION`
- 有意义的变更请追加到 `docs/dev-log.md`

## 6. 报告 Bug

请在 Issue 中提供：

1. FlowX 版本（`git rev-parse HEAD` 或 tag）
2. 运行环境（操作系统、Go 版本、执行后端类型）
3. 复现步骤，**附上最小可复现的流水线 YAML**
4. 实际行为 vs 期望行为
5. 相关日志（注意脱敏）

## 7. 行为准则

请保持专业与尊重。技术分歧请围绕事实与权衡展开，不要针对个人。
维护者保留关闭低质量、偏离项目方向或违反本指南的 PR/Issue 的权利。

## 8. 许可证

本项目以 **GPL-3.0-or-later** 发布（见 [`LICENSE`](./LICENSE)）。
提交贡献即表示你同意你的贡献按 `LICENSE` 及 [`CLA.md`](./CLA.md) 的条款授权。

---

## English Summary

FlowX is **dual-licensed** (GPL-3.0-or-later for the community, plus a separate
commercial license). To keep this model viable, **every contribution requires a
signed [CLA](./CLA.md)**. You keep your copyright — the CLA only grants the
Project Owner the right to relicense and sublicense your contribution under any
terms, including proprietary ones.

Comment `I have read the CLA Document and I hereby sign the CLA` on your first
pull request to sign electronically. If your contribution is a work made for
hire, obtain written authorization from your employer first.

Use [Conventional Commits](https://www.conventionalcommits.org/), run
`gofmt` and `go test ./...` before submitting, and keep PRs focused on a single
change.
