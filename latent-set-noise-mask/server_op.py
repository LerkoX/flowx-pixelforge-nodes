"""COND / LATENT 张量工具算子（ComfyUI 对标第二/三档，插件化交付）。

- cond.combine / cond.average：conditioning 张量拼接/加权（分区构图基础）
- cond.set_area：区域条件（第三档 M4，采样循环内分段前向按区域混合）
- latent.upscale / latent.composite：latent 空间插值放大/贴入拼接
- latent.set_noise_mask：latent 包裹 noise_mask（局部重绘核心，采样循环消费）

分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py
分发为各成员节点的 server_op.py，节点运行时经 ensure_plugin 自注册到推理服务
（POST /admin/plugins，hash 幂等）。使用：对应专属节点或 inference-op 通用节点。

COND 形态：
- SD1.x：张量 (b, seq, dim)，或 cond.set_area 产物的段列表
  [(tensor, area|None, strength), ...]（area 为 (ly,lx,lh,lw) latent 像素坐标）
- SDXL（M5）：SDXLCond（embeds + pooled）/ SDXLSegments（段列表 + 每段 pooled）
  ——本插件对这些算子做保 pooled 的组合（见各函数注释）
- SD3：dict（embeds+pooled，由 sd3-* 插件算子/采样器自成体系，本文件明确报错）
LATENT 为 4D 张量 (b, 4, h/8, w/8) 或 LatentBundle（samples+noise_mask）；
像素坐标参数一律按图像像素计（内部 //8）。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "9e71759f257f"
OP_FAMILY = "cond_latent"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

import torch
from PIL import Image

from app import ops as _core_ops  # 插件可见性约定：可复用核心原语

_SDXL_CONDS = (_core_ops.SDXLCond, _core_ops.SDXLSegments)


# ===========================================================================
# COND 类型层次（对标 ComfyUI conds.py 的 CONDRegular / CONDNoiseShape /
# CONDCrossAttn / CONDList）
#
# 设计目标：让「多条件 + 区域 + 时间窗 + 不同序列长」可任意组合，而不用为
# 每种组合单写一个算子。协议三条：
#   to_segments()  → 统一出口 [(tensor, area, strength, start, end), ...]
#   can_concat(o)  → 协商能否拼接（不同序列长时用 repeat 到 lcm 而不是补零）
#   concat(os)     → 实际拼接
#
# **向后兼容**：裸张量 / 3&5 元组段列表 / SDXLCond / SDXLSegments 继续被
# 服务端 as_cond_segments 直接接受；本层次只是新增一条更结构化的路径。
# ===========================================================================

import math as _math

MAX_CONCAT_REPEAT = 4   # 序列长差异超过 4 倍就不拼接（ComfyUI 的任意上限）


class Conditioning:
    """COND 基类：所有形态都能归一化成段列表。"""

    def to_segments(self):
        raise NotImplementedError

    def can_concat(self, other):
        return False

    def concat(self, others):
        raise NotImplementedError

    def size(self):
        segs = self.to_segments()
        return list(segs[0][0].shape) if segs else []


class CondTensor(Conditioning):
    """单条整图条件（对标 CONDRegular/CONDNoiseShape）。

    area == None 表示全图生效；area=(ly,lx,lh,lw) 表示仅该 latent 区域。
    start/end 为采样进度窗口（对齐 ConditioningSetTimestepRange）。
    """

    def __init__(self, tensor, area=None, strength=1.0,
                 start=0.0, end=1.0):
        self.tensor = tensor
        self.area = area
        self.strength = float(strength)
        self.start = float(start)
        self.end = float(end)

    def to_segments(self):
        return [(self.tensor, self.area, self.strength,
                 self.start, self.end)]

    def with_area(self, area, strength=None):
        return CondTensor(self.tensor, area,
                          self.strength if strength is None else strength,
                          self.start, self.end)

    def with_range(self, start, end):
        return CondTensor(self.tensor, self.area, self.strength, start, end)

    def can_concat(self, other):
        if not isinstance(other, CondTensor):
            return False
        s1, s2 = self.tensor.shape, other.tensor.shape
        if s1[0] != s2[0] or s1[2] != s2[2]:
            return False          # batch / 特征维必须一致
        if s1[1] == s2[1]:
            return True
        diff = _math.lcm(s1[1], s2[1]) // min(s1[1], s2[1])
        return diff <= MAX_CONCAT_REPEAT   # 拉得太远就不拼（性能会崩）

    def concat(self, others):
        """沿序列维拼接（对标 CONDCrossAttn.concat）。

        **关键**：序列长不同时用 `repeat` 填到最小公倍数，而不是补零——
        repeat 不改变注意力语义（同样的 token 重复出现），补零则会把
        padding 位置的注意力权重分走。
        """
        conds = [self.tensor] + [o.tensor for o in others]
        lcm = 1
        for c in conds:
            lcm = _math.lcm(lcm, c.shape[1])
        out = []
        for c in conds:
            if c.shape[1] < lcm:
                c = c.repeat(1, lcm // c.shape[1], 1)
            out.append(c)
        # 拼接后 as 一个整图段：区域/窗口信息在多段场景下逐段保留更准确，
        # 故默认返回段列表而非单张量（调用方按需选 concat / to_segments）
        return torch.cat(out, dim=1)


class ConditioningList(Conditioning):
    """多条件组合容器（对标 CONDList）：逐段保留区域/权重/时间窗。"""

    def __init__(self, items=None):
        self.items = list(items or [])

    def to_segments(self):
        out = []
        for it in self.items:
            if isinstance(it, Conditioning):
                out.extend(it.to_segments())
            elif isinstance(it, list):
                out.extend(it)
            else:
                out.append(it)
        return out

    def can_concat(self, other):
        return isinstance(other, (ConditioningList, CondTensor))

    def concat(self, others):
        items = list(self.items)
        for o in (others if isinstance(others, list) else [others]):
            if isinstance(o, ConditioningList):
                items.extend(o.items)
            else:
                items.append(o)
        return ConditioningList(items)

    def add(self, other):
        return self.concat(other)


def _is_sdxl(cond):
    return isinstance(cond, _SDXL_CONDS)


def _check_same_arch(a, b, op_name):
    """SDXL cond 与 SD1.x cond 不能混用（pooled 缺失会让采样期报错，这里早失败）。"""
    if _is_sdxl(a) != _is_sdxl(b):
        raise ValueError(
            f"{op_name}: 不能混用 SDXL cond 与 SD1.x cond"
            f"（got {type(a).__name__} vs {type(b).__name__}）")


def _as_tensor_cond(cond, op_name):
    """COND 统一为张量；SD3 dict 形式与段列表明确拒绝。
    SDXLCond 返回其 embeds（调用方若需 pooled 请先用 _is_sdxl 分派）。"""
    if isinstance(cond, _core_ops.SDXLCond):
        return cond.embeds
    if isinstance(cond, dict):
        raise ValueError(
            f"{op_name} 暂不支持 SD3 dict 形式 COND（embeds+pooled），"
            f"请使用 sd3-* 节点的配套算子")
    if isinstance(cond, list):
        raise ValueError(
            f"{op_name} 只接受整图 cond 张量，got cond.set_area 段列表")
    if not torch.is_tensor(cond):
        raise ValueError(f"{op_name}: COND 需为张量，got {type(cond).__name__}")
    return cond


def _broadcast_batch(a, b, op_name):
    """batch 维对齐（一方为 1 则 expand）；无法对齐报错。用于 embeds/pooled/段张量。"""
    if a.shape[0] != b.shape[0]:
        if a.shape[0] == 1:
            a = a.expand(b.shape[0], *a.shape[1:])
        elif b.shape[0] == 1:
            b = b.expand(a.shape[0], *b.shape[1:])
        else:
            raise ValueError(
                f"{op_name}: batch 不匹配 {tuple(a.shape)} vs {tuple(b.shape)}")
    return a, b


def _sdxl_size(a, b):
    """合并两 SDXL cond 的原始尺寸：优先 a 的非 0 值，否则 b（0=auto）。"""
    wa, ha = _core_ops.cond_size(a)
    wb, hb = _core_ops.cond_size(b)
    return (wa or wb, ha or hb)


def cond_combine(cond_a, cond_b, pooled_mode="mean"):
    """Conditioning Combine：沿序列维拼接两段 conditioning（对标 ComfyUI
    ConditioningCombine），语义 = 两段提示词同时生效（不受 77 token 截断限制）。
    batch 维不同时按广播对齐（一方为 1 则 expand）。
    任一侧为 cond.set_area 段列表时退化为段列表拼接（各自区域/strength 保留，
    多区域构图 = 多个 set_area 产物经本算子串联）。
    SDXL：embeds 序列拼接，pooled 用 ``pooled_mode`` 决定——
    - ``mean``（默认）：两段均值。单张量 COND 只能承载一个 pooled，均值是
      不丢信息的「同时生效」近似（两段风格各占一半）
    - ``first`` / ``last``：只取前/后一段的 pooled（后段通常承载主体描述）
    - ``segments``：**不复用单张量**，改走 SDXLSegments 逐段前向——每段自带
      pooled、各自前向后再相加，语义最准（= 两条提示词独立生效），代价是
      前向次数翻倍。实测发现均值导致风格漂移时选它。
    注：ComfyUI 的 ConditioningCombine 是 batch 维拼接（同样两次前向），
    本算子默认走序列拼接（一次前向 + 更长的提示词），性能更优。"""
    if _is_sdxl(cond_a) or _is_sdxl(cond_b):
        _check_same_arch(cond_a, cond_b, "cond.combine")
        if isinstance(cond_a, _core_ops.SDXLSegments) \
                or isinstance(cond_b, _core_ops.SDXLSegments):
            # 段列表：逐段保留各自 pooled（与 ComfyUI 行为一致）
            segs = _core_ops.as_cond_segments(cond_a, "cond.combine") \
                + _core_ops.as_cond_segments(cond_b, "cond.combine")
            pooleds = (_core_ops.cond_pooleds(cond_a) or [None] * len(
                _core_ops.as_cond_segments(cond_a, "cond.combine"))) \
                + (_core_ops.cond_pooleds(cond_b) or [None] * len(
                    _core_ops.as_cond_segments(cond_b, "cond.combine")))
            out = _core_ops.SDXLSegments(segs, pooleds, *_sdxl_size(cond_a, cond_b))
            print(f"[cond.combine] SDXL 段列表拼接 -> {len(out)} 段", flush=True)
            return {"cond": out}
        mode = str(pooled_mode or "mean").lower()
        if mode not in ("mean", "first", "last", "segments"):
            raise ValueError(
                f"cond.combine: pooled_mode 需为 mean/first/last/segments，"
                f"got {pooled_mode}")
        a_emb, b_emb = _broadcast_batch(cond_a.embeds, cond_b.embeds, "cond.combine")
        a_pool, b_pool = _broadcast_batch(cond_a.pooled, cond_b.pooled, "cond.combine")
        if mode == "segments":
            # 逐段前向：每段自带 pooled，语义最准（等价两条提示词独立生效）
            pa = a_pool if a_pool.shape[0] > 1 else a_pool.expand(
                a_emb.shape[0], -1)
            pb = b_pool if b_pool.shape[0] > 1 else b_pool.expand(
                b_emb.shape[0], -1)
            segs = [(a_emb, None, 1.0, 0.0, 1.0), (b_emb, None, 1.0, 0.0, 1.0)]
            out = _core_ops.SDXLSegments(
                segs, [pa[:1], pb[:1]], *_sdxl_size(cond_a, cond_b))
            print(f"[cond.combine] SDXL → 2 段逐段前向（pooled 各自生效）",
                  flush=True)
            return {"cond": out}
        if mode == "first":
            pool = a_pool
        elif mode == "last":
            pool = b_pool
        else:
            pool = (a_pool + b_pool) / 2.0
        out = _core_ops.SDXLCond(torch.cat([a_emb, b_emb], dim=1), pool,
                                 *_sdxl_size(cond_a, cond_b))
        print(f"[cond.combine] SDXL {tuple(a_emb.shape)} + {tuple(b_emb.shape)} "
              f"-> {tuple(out.embeds.shape)}（pooled={mode}）", flush=True)
        return {"cond": out}
    if isinstance(cond_a, list) or isinstance(cond_b, list):
        segs = _core_ops.as_cond_segments(cond_a, "cond.combine") \
            + _core_ops.as_cond_segments(cond_b, "cond.combine")
        print(f"[cond.combine] 段列表拼接 -> {len(segs)} 段", flush=True)
        return {"cond": segs}
    a = _as_tensor_cond(cond_a, "cond.combine")
    b = _as_tensor_cond(cond_b, "cond.combine")
    a, b = _broadcast_batch(a, b, "cond.combine")
    out = torch.cat([a, b], dim=1)
    print(f"[cond.combine] {tuple(a.shape)} + {tuple(b.shape)} -> {tuple(out.shape)}",
          flush=True)
    return {"cond": out}


def _resolve_area(x, y, width, height, strength):
    """set_area 参数校验 + 像素坐标→latent 坐标（//8），SD1.x/SDXL 共用。"""
    if width <= 0 or height <= 0:
        raise ValueError(f"cond.set_area: width/height 需 > 0，got ({width},{height})")
    if x < 0 or y < 0:
        raise ValueError(f"cond.set_area: x/y 需 >= 0，got ({x},{y})")
    if strength <= 0:
        raise ValueError(f"cond.set_area: strength 需 > 0，got {strength}")
    area = (y // 8, x // 8, height // 8, width // 8)
    if area[2] <= 0 or area[3] <= 0:
        raise ValueError(
            f"cond.set_area: 区域小于 1 latent px（{width}x{height} 图像像素）")
    return area


def cond_set_area(cond, x=0, y=0, width=512, height=512, strength=1.0):
    """Conditioning Set Area：把整图 cond 标记为只在指定区域生效（对标 ComfyUI
    ConditioningSetArea），输出段列表供 sample 分段前向按区域混合。
    x/y/width/height 为图像像素（内部 //8）；strength 为该段权重（>0）。
    多区域构图：每段 sd.clip.encode 分别 set_area 后经 cond.combine 拼接；
    混合为求和语义（不归一化，重叠区域影响叠加，与 ComfyUI 一致）。
    SDXL：输出 SDXLSegments（每段带自己的 pooled），采样循环逐段前向时
    各自带 text_embeds（与 ComfyUI 每条 conditioning 带 pooled 同语义）。"""
    if isinstance(cond, _core_ops.SDXLCond):
        area = _resolve_area(x, y, width, height, strength)
        out = _core_ops.SDXLSegments([(cond.embeds, area, float(strength))],
                                     [cond.pooled], cond.width, cond.height)
        print(f"[cond.set_area] SDXL area(latent px)={area} strength={strength} "
              f"pooled={tuple(cond.pooled.shape)}", flush=True)
        return {"cond": out}
    t = _as_tensor_cond(cond, "cond.set_area")  # SDXLSegments/段列表在此拒绕
    area = _resolve_area(x, y, width, height, strength)
    print(f"[cond.set_area] area(latent px)={area} strength={strength}", flush=True)
    return {"cond": [(t, area, float(strength))]}


def cond_set_timestep_range(cond, start_percent=0.0, end_percent=1.0):
    """Conditioning Set Timestep Range：条件只在采样进度 [start,end) 内生效
    （对标 ComfyUI ConditioningSetTimestepRange）。

    用途：分段提示词（前 40% 步按"起稿"描述、后 60% 步按"细节"描述）、
    分阶段 ControlNet 风格约束等。采样循环按当前步进度过滤，区间外**整段
    跳过**——既避免不该生效的条件污染画面，也省掉一次前向（速度更快）。

    输入：整图 cond 或 cond.set_area 的段列表（每段统一追加窗口）。
    输出：段列表，每段 (tensor, area, strength, start, end)。
    """
    if not 0.0 <= start_percent <= 1.0 or not 0.0 <= end_percent <= 1.0:
        raise ValueError(
            f"cond.set_timestep_range: start/end 需在 [0,1]，"
            f"got ({start_percent},{end_percent})")
    if end_percent <= start_percent:
        raise ValueError(
            f"cond.set_timestep_range: end 需 > start，"
            f"got ({start_percent},{end_percent})")

    def _wrap(seg):
        t, area = seg[0], seg[1]
        strength = seg[2] if len(seg) > 2 else 1.0
        return (t, area, float(strength), float(start_percent),
                float(end_percent))

    if isinstance(cond, _core_ops.SDXLCond):
        out = [_wrap((cond.embeds, None, 1.0))]
        print(f"[cond.set_timestep_range] SDXL window="
              f"[{start_percent},{end_percent})", flush=True)
        return {"cond": out}
    if isinstance(cond, list):
        out = []
        for seg in cond:
            if not (isinstance(seg, (list, tuple)) and len(seg) in (3, 5)):
                raise ValueError(
                    "cond.set_timestep_range: 段需为 (tensor, area, strength"
                    "[, start, end])")
            out.append(_wrap(seg))
        print(f"[cond.set_timestep_range] {len(out)} 段 window="
              f"[{start_percent},{end_percent})", flush=True)
        return {"cond": out}
    t = _as_tensor_cond(cond, "cond.set_timestep_range")
    print(f"[cond.set_timestep_range] window="
          f"[{start_percent},{end_percent})", flush=True)
    return {"cond": [_wrap((t, None, 1.0))]}


def cond_pack(cond, weight=1.0, start_percent=0.0, end_percent=1.0,
              layer_start=0, layer_end=-1):
    """把任意形态 COND 打包成结构化 Conditioning（复杂编排的起点）。

    对标 ComfyUI 的 cond 包装链：一条裸 cond 依次可加
    - weight：整体权重（等价 ConditioningSetAreaStrength 的全局版）
    - [start_percent, end_percent)：生效的采样进度窗口
    - [layer_start, layer_end)：生效的 cross-attention 层区间
      （对标 CONDCrossAttn：不同层喂不同提示词，例如浅层管构图、深层管细节）

    输出 ConditioningList（服务端 to_segments 归一化），可继续与
    cond.concat / cond.set_area 组合。
    """
    segs = (cond.to_segments() if isinstance(cond, Conditioning)
            else cond if isinstance(cond, list)
            else [(cond, None, 1.0, 0.0, 1.0)])
    items = []
    for seg in segs:
        t = seg[0]
        area = seg[1] if len(seg) > 1 else None
        s0 = float(seg[2]) if len(seg) > 2 else 1.0
        items.append({"tensor": t, "area": area,
                      "strength": s0 * float(weight),
                      "start": float(start_percent),
                      "end": float(end_percent),
                      "layer_start": int(layer_start),
                      "layer_end": int(layer_end)})
    print(f"[cond.pack] {len(items)} 段 weight={weight} "
          f"window=[{start_percent},{end_percent}) "
          f"layers=[{layer_start},{layer_end})", flush=True)
    return {"cond": {"__cond_pack__": items}}


def cond_concat(cond_a, cond_b):
    """多条件拼接（对标 ComfyUI ConditioningConcat / CONDList 协商）。

    与 cond.combine 的差异：
    - cond.combine：**序列维拼接**成一条更长的提示词（突破 77 token）
    - cond.concat ：**并列多条条件**，各自保留区域/权重/时间窗/层区间
      （多区域构图、分段提示词、分阶段约束的标准做法）
    序列长不一致时用 repeat 到 lcm 填充（不改变注意力语义，优于补零）。
    """
    a = cond_a if isinstance(cond_a, Conditioning) else ConditioningList([cond_a])
    b = cond_b if isinstance(cond_b, Conditioning) else ConditioningList([cond_b])
    out = a.concat(b)
    print(f"[cond.concat] -> {len(out.to_segments())} 段", flush=True)
    return {"cond": out}


def cond_average(cond_a, cond_b, weight=0.5):
    """Conditioning Average：两段 conditioning 按权重加权平均（对标 ComfyUI
    ConditioningAverage）：out = a*weight + b*(1-weight)。
    序列长度不同时短者补零到长者（dim=1）。
    SDXL：embeds 与 pooled 同时加权平均（pooled 为全局偏置，权重语义一致）。"""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"cond.average: weight 需在 [0,1]，got {weight}")
    if _is_sdxl(cond_a) or _is_sdxl(cond_b):
        _check_same_arch(cond_a, cond_b, "cond.average")
        if isinstance(cond_a, _core_ops.SDXLSegments) \
                or isinstance(cond_b, _core_ops.SDXLSegments):
            raise ValueError(
                "cond.average: 暂不支持分段（set_area 产物）cond 的加权平均，"
                "请先 combine 或用整图 cond")
        a = cond_a.embeds
        b = cond_b.embeds
        pa, pb = _broadcast_batch(cond_a.pooled, cond_b.pooled, "cond.average")
        a, b = _broadcast_batch(a, b, "cond.average")
        a, b = _pad_seq(a, b)
        out = _core_ops.SDXLCond(a * weight + b * (1.0 - weight),
                                pa * weight + pb * (1.0 - weight),
                                *_sdxl_size(cond_a, cond_b))
        print(f"[cond.average] SDXL weight={weight} "
              f"-> embeds{tuple(out.embeds.shape)} pooled{tuple(out.pooled.shape)}",
              flush=True)
        return {"cond": out}
    a = _as_tensor_cond(cond_a, "cond.average")
    b = _as_tensor_cond(cond_b, "cond.average")
    a, b = _broadcast_batch(a, b, "cond.average")
    a, b = _pad_seq(a, b)
    out = a * weight + b * (1.0 - weight)
    print(f"[cond.average] weight={weight} -> {tuple(out.shape)}", flush=True)
    return {"cond": out}


def _pad_seq(a, b):
    """序列长度不同时用 **repeat 到最小公倍数** 对齐（dim=1）。

    【为什么不是补零】早期实现是把短的一方补零到长者，但 padding 位置会
    分走 cross-attention 的权重——等于稀释真实 token 的注意力（提示词
    越长越明显）。ComfyUI 的 CONDCrossAttn.concat 用
    `c.repeat(1, lcm // c.shape[1], 1)`：同一个 token 重复出现，注意力语义
    不变（重复 token 的关注度均摊后总量相当），是目前公认的稳妥做法。
    （参考 comfy/conds.py:CONDCrossAttn.concat）

    长度差超过 MAX_CONCAT_REPEAT 倍时退化回补零——repeat 太多次会让序列
    膨胀到几倍长，注意力开销按 O(n²) 涨，得不偿失。
    """
    na, nb = a.shape[1], b.shape[1]
    if na == nb:
        return a, b
    lcm = _math.lcm(na, nb)
    if lcm // min(na, nb) <= MAX_CONCAT_REPEAT:
        if na < lcm:
            a = a.repeat(1, lcm // na, 1)
        if nb < lcm:
            b = b.repeat(1, lcm // nb, 1)
        return a, b
    # 差异过大：退回补零（避免序列膨胀）
    n = max(na, nb)
    if na < n:
        a = torch.cat([a, a.new_zeros(a.shape[0], n - na, a.shape[2])], dim=1)
    if nb < n:
        b = torch.cat([b, b.new_zeros(b.shape[0], n - nb, b.shape[2])], dim=1)
    return a, b


_INTERP = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic"}


def latent_upscale(latent, scale=2.0, width=0, height=0, method="bicubic"):
    """Latent Upscale：latent 空间插值放大（对标 ComfyUI LatentUpscale）。
    尺寸二选一：width/height（图像像素，内部 //8）均 >0 时按目标尺寸，
    否则按 scale 倍率；latent 尺寸天然整数对齐（1 latent px = 8 图像 px）。
    method 支持 bicubic（默认）/bilinear/nearest。
    LatentBundle（带 noise_mask）明确拒绝：请先放大再 set_noise_mask。"""
    if isinstance(latent, _core_ops.LatentBundle):
        raise ValueError(
            "latent.upscale 不接受带 noise_mask 的 latent（mask 不会随放大缩放）；"
            "请先 latent.upscale 再 latent.set_noise_mask")
    if method not in _INTERP:
        raise ValueError(
            f"unknown method '{method}', available: {sorted(_INTERP)}")
    if not torch.is_tensor(latent) or latent.dim() != 4:
        raise ValueError(
            f"latent.upscale: 需 4D latent 张量，got "
            f"{tuple(latent.shape) if torch.is_tensor(latent) else type(latent).__name__}")
    b, c, lh, lw = latent.shape
    if width > 0 and height > 0:
        th, tw = max(1, height // 8), max(1, width // 8)
    else:
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        th, tw = max(1, round(lh * scale)), max(1, round(lw * scale))
    kwargs = {"mode": _INTERP[method]}
    if _INTERP[method] in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    out = torch.nn.functional.interpolate(latent, size=(th, tw), **kwargs)
    print(f"[latent.upscale] {tuple(latent.shape)} -> {tuple(out.shape)} "
          f"({method})", flush=True)
    return {"latent": out}


def latent_composite(dst, src, x=0, y=0, feather=0):
    """Latent Composite：把 src latent 贴入 dst 的 (x, y) 处（对标 ComfyUI
    LatentComposite），x/y 为图像像素坐标（内部 //8）。越界部分自动裁剪；
    feather>0（图像像素）时边缘线性羽化过渡。batch：src 为 1 时广播到 dst。"""
    for name, t in (("dst", dst), ("src", src)):
        if isinstance(t, _core_ops.LatentBundle):
            raise ValueError(
                f"latent.composite: {name} 带 noise_mask（不会随贴入变换）；"
                f"请先 composite 再 latent.set_noise_mask")
        if not torch.is_tensor(t) or t.dim() != 4:
            raise ValueError(f"latent.composite: {name} 需 4D latent 张量")
    if x < 0 or y < 0:
        raise ValueError(f"latent.composite: x/y 需 >= 0，got ({x},{y})")
    lx, ly = x // 8, y // 8
    b, c, dh, dw = dst.shape
    h = min(src.shape[2], dh - ly)
    w = min(src.shape[3], dw - lx)
    if h <= 0 or w <= 0:
        raise ValueError(
            f"latent.composite: src 完全越界（dst {dh}x{dw} latent px，"
            f"起点 ({lx},{ly})）")
    src_c = src[:, :, :h, :w].to(dst.dtype)  # 原地运算前统一 dtype（fp16/fp32 混用）
    if src_c.shape[0] == 1 and b > 1:
        src_c = src_c.expand(b, -1, -1, -1)
    elif src_c.shape[0] != b:
        raise ValueError(
            f"latent.composite: batch 不匹配 dst={b} src={src_c.shape[0]}")
    out = dst.clone()
    region = out[:, :, ly:ly + h, lx:lx + w]
    f = feather // 8
    if f > 0 and h > 2 * f and w > 2 * f:
        # 边缘线性斜坡羽化（1 内部 → 0 边缘），x/y 两维斜坡相乘
        ramp_y = torch.ones(h, device=dst.device, dtype=dst.dtype)
        ramp_x = torch.ones(w, device=dst.device, dtype=dst.dtype)
        for i in range(f):
            v = (i + 1) / (f + 1)
            ramp_y[i] = ramp_y[h - 1 - i] = v
            ramp_x[i] = ramp_x[w - 1 - i] = v
        mask = (ramp_y[:, None] * ramp_x[None, :])[None, None]
        region.mul_(1 - mask).add_(src_c * mask)
    else:
        region.copy_(src_c)
    print(f"[latent.composite] src {tuple(src_c.shape)} -> ({lx},{ly}) "
          f"feather={f} latent px", flush=True)
    return {"latent": out}


def latent_set_noise_mask(latent, mask):
    """Set Latent Noise Mask：给 latent 包裹 noise_mask（对标 ComfyUI
    SetLatentNoiseMask），局部重绘核心——sample 循环内每步把 mask 外区域
    混回原始 latent 的当步加噪版，mask 内正常去噪。
    mask 为灰度 IMAGE 对象（对齐第一档蒙版约定：白=重绘区域），
    内部 resize 到 latent 尺寸转 0..1 张量。重复包裹明确拒绝。"""
    samples, existing = _core_ops.resolve_latent(latent)
    if existing is not None:
        raise ValueError("latent.set_noise_mask: 已有 noise_mask，请勿重复包裹")
    if not torch.is_tensor(samples) or samples.dim() != 4:
        raise ValueError("latent.set_noise_mask: latent 需 4D 张量")
    if isinstance(mask, list):
        if len(mask) != 1:
            raise ValueError(
                f"latent.set_noise_mask: mask 需单张灰度 IMAGE，got {len(mask)} 张")
        mask = mask[0]
    if not isinstance(mask, Image.Image):
        raise ValueError(
            f"latent.set_noise_mask: mask 需为 IMAGE 对象（灰度），"
            f"got {type(mask).__name__}")
    lh, lw = samples.shape[2], samples.shape[3]
    m = mask.convert("L").resize((lw, lh), Image.BILINEAR)
    import numpy as np  # 懒加载：与核心 ops 同约定（无 GPU 契约测试可导入）
    arr = np.asarray(m, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None].to(device=samples.device,
                                             dtype=samples.dtype)
    print(f"[latent.set_noise_mask] mask -> (1,1,{lh},{lw}) "
          f"mean={float(t.mean()):.3f}", flush=True)
    return {"latent": _core_ops.LatentBundle(samples, t)}


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "cond.combine",
        inputs={"cond_a": "COND", "cond_b": "COND",
                "pooled_mode": "STRING"},
        outputs={"cond": "COND"},
        description="Conditioning Combine：两段 conditioning 沿序列维拼接，"
                    "两段提示词同时生效（可突破 77 token 截断）；"
                    "暂不支持 SD3 dict 形式 COND")(cond_combine)
    registry.register(
        "cond.average",
        inputs={"cond_a": "COND", "cond_b": "COND", "weight": "FLOAT"},
        outputs={"cond": "COND"},
        description="Conditioning Average：out = cond_a*weight + cond_b*(1-weight)，"
                    "weight 默认 0.5；序列长度不同时短者补零；"
                    "暂不支持 SD3 dict 形式 COND")(cond_average)
    registry.register(
        "latent.upscale",
        inputs={"latent": "LATENT", "scale": "FLOAT", "width": "INT",
                "height": "INT", "method": "STRING"},
        outputs={"latent": "LATENT"},
        description="Latent Upscale：latent 空间插值放大，按 scale 倍率（默认 2.0）"
                    "或目标 width/height（图像像素）；method 支持 "
                    "bicubic/bilinear/nearest")(latent_upscale)
    registry.register(
        "latent.composite",
        inputs={"dst": "LATENT", "src": "LATENT", "x": "INT", "y": "INT",
                "feather": "INT"},
        outputs={"latent": "LATENT"},
        description="Latent Composite：把 src 贴入 dst 的 (x,y) 处（图像像素坐标），"
                    "越界自动裁剪；feather>0（图像像素）边缘线性羽化；"
                    "带 noise_mask 的 latent 明确拒绝（先 composite 再 set_noise_mask）")(latent_composite)
    registry.register(
        "cond.set_area",
        inputs={"cond": "COND", "x": "INT", "y": "INT", "width": "INT",
                "height": "INT", "strength": "FLOAT"},
        outputs={"cond": "COND"},
        description="Conditioning Set Area：cond 标记为只在 (x,y,width,height) 区域"
                    "生效（图像像素），strength 为该段权重（默认 1.0）；"
                    "多区域 = 多个 set_area 经 cond.combine 拼接；"
                    "暂不与 controlnet 组合")(cond_set_area)
    registry.register(
        "cond.pack",
        inputs={"cond": "COND", "weight": "FLOAT",
                "start_percent": "FLOAT", "end_percent": "FLOAT",
                "layer_start": "INT", "layer_end": "INT"},
        outputs={"cond": "COND"},
        description="把 COND 打包成结构化条件：整体权重 + 采样进度窗口 + "
                    "cross-attention 层区间（对标 ComfyUI cond 包装链）。"
                    "复杂多条件编排的起点，可继续 cond.concat / cond.set_area")(cond_pack)
    registry.register(
        "cond.concat",
        inputs={"cond_a": "COND", "cond_b": "COND"},
        outputs={"cond": "COND"},
        description="多条件并列拼接（对标 ConditioningConcat/CONDList 协商）："
                    "各条条件保留自己的区域/权重/时间窗/层区间。与 cond.combine "
                    "的序列维拼接不同，本算子是「多条并存」")(cond_concat)
    registry.register(
        "cond.set_timestep_range",
        inputs={"cond": "COND", "start_percent": "FLOAT",
                "end_percent": "FLOAT"},
        outputs={"cond": "COND"},
        description="Conditioning Set Timestep Range：条件只在采样进度 "
                    "[start,end) 内生效（对标 ComfyUI ConditioningSet"
                    "TimestepRange）。分段提示词/分阶段约束；区间外整段跳过"
                    "（避免污染 + 省一次前向）。可叠加 cond.set_area")(cond_set_timestep_range)
    registry.register(
        "latent.set_noise_mask",
        inputs={"latent": "LATENT", "mask": "IMAGE"},
        outputs={"latent": "LATENT"},
        description="Set Latent Noise Mask：latent 包裹 noise_mask（局部重绘核心）；"
                    "mask 为灰度 IMAGE（白=重绘区域），内部 resize 到 latent 尺寸；"
                    "配 sd.vae.encode + sd.sample(denoise<1) 做 inpaint")(latent_set_noise_mask)
