#!/usr/bin/env python3
"""image-montage 本机契约单测（不依赖 GPU/推理服务，PIL 即可）。

重点：**尺寸公式**（复核脚本按它对表）、自动列数、cell 自动取最大尺寸、
fit/cover 语义、标签行高、以及"无缩放时格内像素与源逐字节一致"（A4 判据）。
用法：python3 nodes/image-montage/test_main.py
"""
import importlib.util
import io
import pathlib
import sys
import types as _types

from PIL import Image, ImageChops

HERE = pathlib.Path(__file__).parent
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


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    fc = _types.ModuleType("flowx_client")
    fc.get_bytes = lambda *a, **k: b""
    fc.post_bytes = lambda *a, **k: {"id": "obj"}
    sys.modules["flowx_client"] = fc
    sys.path.insert(0, str(HERE))
    ib = load_mod("image_batch", HERE / "image_batch.py")
    im = load_mod("image_montage", HERE / "image_montage.py")

    # 造 3 张不同尺寸、颜色各异的图；montage 用 get_pimg_bytes 取数（打补丁到模块名）
    src = {}
    for i, (w, h, c) in enumerate(((40, 30, (200, 0, 0)), (60, 20, (0, 200, 0)),
                                   (30, 50, (0, 0, 200)))):
        b = io.BytesIO()
        Image.new("RGB", (w, h), c).save(b, format="PNG")
        src[f"id{i}"] = b.getvalue()
    im.get_pimg_bytes = lambda url, oid, tok=None: src[oid]

    check("auto_cols(3) == 2", im.auto_cols(3) == 2)
    check("auto_cols(4) == 2", im.auto_cols(4) == 2)
    check("auto_cols(9) == 3", im.auto_cols(9) == 3)
    check("auto_cols(0/1) == 1", im.auto_cols(0) == 1 and im.auto_cols(1) == 1)
    check("颜色解析 #RGB 与 #RRGGBB",
          im._color("#f00") == (255, 0, 0) and im._color("#00ff00") == (0, 255, 0))
    check("颜色非法回落默认", im._color("nope") == (22, 22, 24))

    ids = ["id0", "id1", "id2"]
    pil, grid = im.montage("http://x", ids, None, cols=0, cell_w=0, cell_h=0, pad=10,
                           label="none")
    check("grid JSON 是 @ 信封", grid.startswith("@"), grid)
    cols, rows = ib.loads(grid)
    check("自动列数 2 列 / 2 行", (cols, rows) == (2, 2), grid)
    check("cell 自动取最大宽高（60×50）",
          pil.width == 2 * 60 + 3 * 10, f"{pil.width} vs {2*60+3*10}")
    check("高度 == rows*cell_h + (rows+1)*pad（label=none 时 label_h=0）",
          pil.height == 2 * 50 + 3 * 10, f"{pil.height} vs {2*50+3*10}")
    check("背景色生效（右下空白区是 #161618）",
          pil.getpixel((pil.width - 2, pil.height - 2)) == (22, 22, 24),
          str(pil.getpixel((pil.width - 2, pil.height - 2))))

    # 无缩放（cell == 图尺寸）⇒ 格内像素必须与源逐字节一致
    one, _ = im.montage("http://x", ["id0"], None, cols=1, cell_w=40, cell_h=30, pad=0,
                        label="none")
    check("无缩放时格内与源逐字节一致（A4 判据）",
          ImageChops.difference(one, Image.open(io.BytesIO(src["id0"])).convert("RGB"))
          .getbbox() is None, "diff bbox 应为 None")

    # 标签行高进公式
    lab, _ = im.montage("http://x", ids, None, cols=2, cell_w=60, cell_h=50, pad=10,
                        label="index", label_h=20)
    check("label_h 进高度公式",
          lab.height == 2 * 50 + 3 * 10 + 2 * 20, f"{lab.height}")
    check("label=none 时 label_h 不生效",
          im.montage("http://x", ids, None, cols=2, cell_w=60, cell_h=50, pad=10,
                     label="none", label_h=20)[0].height == 2 * 50 + 3 * 10)

    # fit vs cover
    fit1, _ = im.montage("http://x", ["id2"], None, cols=1, cell_w=60, cell_h=50, pad=0,
                         fit="fit")
    cov1, _ = im.montage("http://x", ["id2"], None, cols=1, cell_w=60, cell_h=50, pad=0,
                         fit="cover")
    check("fit/cover 输出尺寸都 == cell 尺寸", fit1.size == cov1.size == (60, 50))
    check("fit 会露出背景（左边缘是背景色）",
          fit1.getpixel((0, 0)) == (22, 22, 24), str(fit1.getpixel((0, 0))))
    check("cover 铺满（无背景色）", cov1.getpixel((0, 0)) != (22, 22, 24),
          str(cov1.getpixel((0, 0))))

    # names（label=name 用文件名，而不是对象 ID）
    named, _ = im.montage("http://x", ids, None, cols=3, cell_w=60, cell_h=50, pad=0,
                          label="name", label_h=16,
                          names=["a.png", "b.png", "c.png"])
    check("label=name + names 可出图", named.size == (180, 66), str(named.size))
    try:
        im.montage("http://x", ids, None, label="name")
        check("label=name 缺 names 必须报错", False, "未报错")
    except ValueError as e:
        check("label=name 缺 names 报错并指路", "names" in str(e))
    try:
        im.montage("http://x", ids, None, label="name", names=["only-one.png"])
        check("names 数量不符必须报错", False, "未报错")
    except ValueError:
        check("names 数量不符报错", True)

    # 参数校验
    for kwargs, why in ((dict(fit="stretch"), "fit"), (dict(label="bogus"), "label")):
        try:
            im.montage("http://x", ids, None, **kwargs)
            check(f"{why} 非法必须报错", False, "未报错")
        except ValueError:
            check(f"{why} 非法报错", True)
    try:
        im.montage("http://x", [], None)
        check("空列表必须报错", False, "未报错")
    except ValueError:
        check("空列表报错", True)

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
