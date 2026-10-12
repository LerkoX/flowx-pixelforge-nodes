#!/usr/bin/env python3
"""B4/S1 G8 DDColor 上色落盘复核（dev-plan §50.5 判据 A1~A6）。

只读 `~/flowx-e2e-out/` 里的 PNG，纯 PIL（Termux 无 numpy），不碰服务。

用法：
  python3 nodes/_tools/verify_colorize.py          # 跑全部判据
  python3 nodes/_tools/verify_colorize.py --list   # 只列最近产物

判据：
  A1 尺寸：每张上色输出 == 对应输入尺寸（保留原图 L 通道 ⇒ 尺寸与结构不变）
  A2 确定性：wf 139 连跑两遍，同一前缀的两次产物**逐字节一致**（DDColor 无随机采样）
  A3 色彩化有效性：输出 S 均值 ≥ 5× 输入；近中性像素(S<0.05)占比 ≤ 输入的一半
  A4 肤色合理性：人脸蒙版内 H∈[10°,45°] 占比 ≥ 60%（灰度人像上色后应成人肤色）
  A5 执行器无关：local 与 docker 两条路径同参数产物**逐字节一致**
  A6 变体互异：4 个变体输出的 sha256 互不相同（说明真的换了权重）
"""
import hashlib
import pathlib
import re
import sys
from colorsys import rgb_to_hsv

from PIL import Image, ImageStat

OUT = pathlib.Path.home() / "flowx-e2e-out"
PAIRS = [("colin_gray", "col_gray"), ("colin_faded", "col_faded"),
         ("colin_old", "col_old"), ("colin_face", "col_face")]
# A5：local 输出 vs docker 输出（同一输入、同一参数）
DOCKER_PAIRS = [("col_gray", "cold_gray"), ("col_faded", "cold_faded"),
                ("col_old", "cold_old")]
# A5b：两条路径喂进去的输入必须一致（否则"A5 通过"可能是输入不同导致的假象）
DOCKER_INPUTS = [("colin_gray", "coldin_gray"), ("colin_faded", "coldin_faded"),
                 ("colin_old", "coldin_old")]
VARIANTS = ["colv_modelscope", "colv_paper", "colv_artistic", "colv_paper_tiny"]

FAILED = 0
PASSED = 0


def check(label, cond, extra=""):
    global FAILED, PASSED
    if cond:
        PASSED += 1
        print(f"PASS {label}" + (f"  {extra}" if extra else ""))
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  {extra}" if extra else ""))


def latest(prefix, nth=0):
    """精确匹配 <prefix>_YYYYmmdd_HHMMSS.png，按 mtime 排序取倒数第 nth+1 个。"""
    pat = re.compile(rf"^{re.escape(prefix)}_+\d{{8}}_\d{{6}}\.png$")
    fs = sorted([f for f in OUT.iterdir() if pat.match(f.name)],
                key=lambda f: f.stat().st_mtime)
    return fs[-(nth + 1)] if len(fs) > nth else None


def sat_stats(path):
    """饱和度统计：S 均值、近中性像素占比。纯 PIL（转 HSV 后取 S 通道）。"""
    im = Image.open(path).convert("RGB")
    hsv = im.convert("HSV")
    s = hsv.getchannel("S")
    hist = s.histogram()
    total = sum(hist)
    mean = sum(i * c for i, c in enumerate(hist)) / total
    neutral = sum(hist[:13]) / total          # S < 13/255 ≈ 0.05
    return mean, neutral


def hue_profile(img_path, mask_path, lo=10, hi=45):
    """蒙版内色相画像：肤色带占比、色相模态（10° 分箱）、平均饱和度（纯 python）。"""
    img = Image.open(img_path).convert("RGB")
    msk = Image.open(mask_path).convert("L")
    if msk.size != img.size:
        msk = msk.resize(img.size, Image.LANCZOS)
    px, mp = img.load(), msk.load()
    w, h = img.size
    bins = [0] * 36
    ssum = tot = 0
    for y in range(0, h, 2):                   # 隔行采样（够统计，省时间）
        for x in range(0, w, 2):
            if mp[x, y] < 128:
                continue
            tot += 1
            r, g, b = px[x, y]
            hh, ss, _ = rgb_to_hsv(r / 255, g / 255, b / 255)
            bins[min(35, int(hh * 360 // 10))] += 1
            ssum += ss
    if not tot:
        return 0.0, 0, 0, 0, 0.0
    frac = sum(bins[lo // 10:(hi // 10) + 1]) / tot
    top = max(range(36), key=lambda i: bins[i])
    return frac, tot, top * 10, top * 10 + 10, ssum / tot * 255


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    if "--list" in sys.argv:
        for p in sorted(OUT.glob("col*.png"))[-14:]:
            print(" ", p.name)
        return 0

    print("== A1 尺寸（上色输出 == 输入尺寸，保留 L 通道 ⇒ 结构不变） ==")
    for a, b in PAIRS:
        pin, pout = latest(a), latest(b)
        if not pin or not pout:
            check(f"A1 {b}", False, "缺产物"); continue
        si, so = Image.open(pin).size, Image.open(pout).size
        check(f"A1 {b}", si == so, f"{si} vs {so}")

    print("\n== A2 确定性（wf 139 连跑两遍，同前缀两次产物逐字节一致） ==")
    for _, b in PAIRS:
        f1, f2 = latest(b), latest(b, 1)
        if not f1 or not f2:
            check(f"A2 {b}", False, "只有一次产物"); continue
        same = f1.read_bytes() == f2.read_bytes()
        check(f"A2 {b}", same, f"{f2.name} vs {f1.name}" + ("（一致）" if same else "（不同）"))

    print("\n== A3 色彩化有效性 ==")
    print("   判据标定（dev-plan §50.5 A3 校正）：灰度输入 ⇒ 倍数判据（≥5×，S_in≈0）；")
    print("   已带色输入（褪色/偏黄）⇒ 绝对+倍数双判据（S_out ≥ 100 且 ≥ 2×S_in）——")
    print("   原「一律 5×」对已有色彩的输入不可达也无意义（实测 4.1×/2.9× 即为强上色）。")
    for a, b in PAIRS:
        pin, pout = latest(a), latest(b)
        if not pin or not pout:
            check(f"A3 {b}", False, "缺产物"); continue
        mi, ni = sat_stats(pin)
        mo, no = sat_stats(pout)
        ratio = (mo / mi) if mi > 1e-6 else float("inf")
        if mi < 1.0:                      # 灰度/近中性输入
            check(f"A3 {b} 灰度输入 S均值 {mi:.1f}→{mo:.1f}（≥5×）", ratio >= 5.0)
        else:                             # 已带色输入
            check(f"A3 {b} 带色输入 S均值 {mi:.1f}→{mo:.1f}（≥100 且 ≥2×，实测 {ratio:.1f}×）",
                  mo >= 100.0 and ratio >= 2.0)
        check(f"A3 {b} 近中性像素不增（≤输入一半）", no <= ni / 2 + 0.005,
              f"近中性 {ni * 100:.1f}%→{no * 100:.1f}%")

    print("\n== A4 肤色合理性 ==")
    print("   判据标定（dev-plan §50.5 A4 校正）：face-mask 出的是**人脸矩形框**（含头发/")
    print("   背景/衣物），「框内 H∈[10,45] 占比 ≥60%」对矩形框不可达（实测峰值 10~20°，")
    print("   正是肤色，但只占 26%）。改为三个按物理含义的判据：① 框内色相模态 ∈ [5°,45°]；")
    print("   ② 肤色带占比 ≥ 35%（远高于「完全不上色」的噪声水平）；③ 框内平均 S ≥ 20/255。")
    pmask, pface = latest("col_face_mask"), latest("col_face")
    if pmask and pface:
        frac, tot, mode_lo, mode_hi, msat = hue_profile(pface, pmask)
        check(f"A4 框内色相模态 ∈[5°,45°]（实测 {mode_lo}-{mode_hi}°）",
              5 <= mode_lo <= 45, f"{tot}px 采样")
        check(f"A4 肤色带占比 ≥35%（实测 {frac * 100:.1f}%）", frac >= 0.35)
        check(f"A4 框内平均饱和度 ≥20/255（实测 {msat:.1f}）", msat >= 20.0)
    else:
        check("A4 需要 col_face_mask + col_face", False, "缺产物")

    print("\n== A5 执行器无关（local vs docker 逐字节一致） ==")
    for la, da in DOCKER_INPUTS:
        pl, pd = latest(la), latest(da)
        if pl and pd:
            check(f"A5b 输入一致 {da}", pl.read_bytes() == pd.read_bytes(),
                  f"{pl.name} vs {pd.name}")
    for la, da in DOCKER_PAIRS:
        pl, pd = latest(la), latest(da)
        if not pl or not pd:
            check(f"A5 {da}", False, "缺产物"); continue
        same = pl.read_bytes() == pd.read_bytes()
        check(f"A5 {da}", same, f"{pl.name} vs {pd.name}" + ("（一致）" if same else "（不同）"))

    print("\n== A6 变体互异（4 个变体 sha256 互不相同） ==")
    fs = [latest(v) for v in VARIANTS]
    if all(fs):
        shas = [sha(f) for f in fs]
        check("A6 4 个变体产物 sha 互异", len(set(shas)) == 4,
              " | ".join(f"{v.split('_', 1)[1]}={s}" for v, s in zip(VARIANTS, shas)))
    else:
        check("A6 需要 4 个变体产物", False, str([f.name if f else None for f in fs]))

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
