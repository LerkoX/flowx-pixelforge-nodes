#!/usr/bin/env python3
"""B5/52.2 G11 批量节点落盘复核（dev-plan §52.2 判据 A1~A6）。

只读 `~/flowx-e2e-out/` 与 `~/flowx-input/`，纯 PIL + 标准库（Termux 无 numpy），不碰服务。

判据：
  A1 批量另存：文件名 == <prefix>_<序号 3 位>.png，数量与来源一致
  A2 批量另存与来源等价：**逐字节一致**，或（服务端对上传图做了归一化编码时）
     **解码像素逐字节一致** —— 实测 gray_person.png 是灰度模式，上传后服务端重编码
     （37605B → 52387B，像素不变），所以判据必须允许这一档
  A3 拼图尺寸 == 公式 W = cols*cell_w + (cols+1)*pad / H = rows*cell_h + (rows+1)*pad
     + rows*label_h（两组：roundtrip 3 列 1:1、preview 3 列带标签）
  A4 拼图格内与来源**逐字节一致**（cell 尺寸 == 图尺寸、pad=0 ⇒ 必须是 0 差）
  A5 联系表 tile 数与公式自洽（由尺寸与格尺寸反推 cols/rows）
  A6 人眼物料已生成（联系表拷进 ~/flowx-output/）
"""
import hashlib
import pathlib
import re
import sys

from PIL import Image, ImageChops

OUT = pathlib.Path.home() / "flowx-e2e-out"
IN = pathlib.Path.home() / "flowx-input"
MEDIA = pathlib.Path.home() / "flowx-output"
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


def latest(prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_+\d{{8}}_\d{{6}}\.png$")
    fs = sorted([f for f in OUT.iterdir() if pat.match(f.name)],
                key=lambda f: f.stat().st_mtime)
    return fs[-1] if fs else None


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    trio = ["faded.png", "gray_person.png", "old_yellow.png"]   # 与流水线 pattern 同序
    srcs = [IN / n for n in trio]

    print("== A1/A2 批量另存（image-batch-save） ==")
    saved = [OUT / f"batch_rt_{i:03d}.png" for i in (1, 2, 3)]
    check("A1 三个文件都落盘了", all(f.is_file() for f in saved),
          str([f.name for f in saved if f.is_file()]))
    if all(f.is_file() for f in saved):
        byte_exact = pixel_exact = 0
        for s, f in zip(srcs, saved):
            same_bytes = sha(s) == sha(f)
            same_px = ImageChops.difference(Image.open(s).convert("RGB"),
                                            Image.open(f).convert("RGB")).getbbox() is None
            byte_exact += 1 if same_bytes else 0
            pixel_exact += 1 if same_px else 0
            check(f"A2 {f.name} 与来源等价", same_bytes or same_px,
                  ("逐字节一致" if same_bytes
                   else ("像素一致（服务端归一化编码，允许档）" if same_px
                         else f"不一致 {sha(s)[:12]} vs {sha(f)[:12]}")))
        print(f"  [观察] 逐字节一致的占 {byte_exact}/{len(srcs)}，"
              f"解到像素仍一致的占 {pixel_exact}/{len(srcs)}"
              f"（差额来自服务端对上传图的归一化重编码）")

    print("\n== A3/A4/A5 拼图（image-montage） ==")
    grid = latest("batch_grid")
    if grid is None:
        check("需要有 batch_grid 产物", False, "缺产物")
    else:
        g = Image.open(grid).convert("RGB")
        # 流水线参数：cols=3, cell_w=0(=最大宽 512), cell_h=0(=最大高 384), pad=0, label=none
        cols, cw, ch, pad, label_h, rows = 3, 512, 384, 0, 0, 1
        exp = (cols * cw + (cols + 1) * pad,
               rows * ch + (rows + 1) * pad + rows * label_h)
        check(f"A3 尺寸 == 公式 {exp[0]}×{exp[1]}", g.size == exp,
              f"{g.size} vs {exp}")
        check("A5 由尺寸反推 grid == [3, 1]",
              (g.width // cw, g.height // ch) == (cols, rows),
              f"{(g.width // cw, g.height // ch)}")
        same = []
        for i, s in enumerate(srcs):
            cell = g.crop((i * cw, 0, (i + 1) * cw, ch))
            ref = Image.open(s).convert("RGB")
            same.append(ImageChops.difference(cell, ref).getbbox() is None)
        check("A4 三个格内与来源逐字节一致", all(same), str(same))

    print("\n== A3' 带标签联系表（image-batch-load → preview） ==")
    prev = latest("batch_preview")
    if prev is None:
        check("需要有 batch_preview 产物", False, "缺产物")
    else:
        p = Image.open(prev)
        # 流水线参数：cols=0(自动 6 张 → 3 列), cell 320×240, pad=8, label=index, label_h=20
        cols, cw, ch, pad, lh = 3, 320, 240, 8, 20
        rows = 2                                  # 6 张图 / 3 列
        exp = (cols * cw + (cols + 1) * pad,
               rows * ch + (rows + 1) * pad + rows * lh)
        check(f"A3' 尺寸 == 公式 {exp[0]}×{exp[1]}（含 label_h）", p.size == exp,
              f"{p.size} vs {exp}")
        # 标签行区域（每行底部 lh 高）不该是纯背景（说明真的画了标签文字）
        # 注意要**扫一块区域**：默认位图字体的字形上沿可能整行留白（只点一个 y 会假失败）
        y0 = pad + ch + 1
        hit = any(p.getpixel((x, y)) != (22, 22, 24)
                  for y in range(y0, y0 + lh - 2) for x in range(10, 120, 2))
        check("A3' 标签行确实有内容（非纯背景色）", hit, f"扫描 y={y0}..{y0+lh-2}")

    print("\n== A6 人眼物料 ==")
    dst = MEDIA / "b5_batch_preview.png"
    check("A6 联系表已拷进 ~/flowx-output", dst.is_file(),
          f"{dst.name} {dst.stat().st_size // 1024 if dst.is_file() else 0}KB")

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
