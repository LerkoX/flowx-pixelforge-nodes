#!/usr/bin/env python3
"""image-batch-save 本机契约单测（不依赖 GPU/推理服务）。

重点：落盘文件名规则（`<prefix>_<序号>.<ext>`、start_index、zero_pad）、
真实写文件、以及空列表/非法扩展名/目录自动创建。
用法：python3 nodes/image-batch-save/test_main.py
"""
import importlib.util
import io
import pathlib
import sys
import tempfile
import types as _types

from PIL import Image

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


def png_bytes(color=(10, 20, 30)):
    b = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(b, format="PNG")
    return b.getvalue()


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    fc = _types.ModuleType("flowx_client")
    fc.get_bytes = lambda *a, **k: png_bytes()
    fc.post_bytes = lambda *a, **k: {"id": "obj"}
    sys.modules["flowx_client"] = fc
    sys.path.insert(0, str(HERE))
    ib = load_mod("image_batch", HERE / "image_batch.py")

    check("load_ids 解析信封", ib.load_ids('@["a","b"]') == ["a", "b"])
    check("load_ids 容忍逗号分隔", ib.load_ids("a,b") == ["a", "b"])
    check("load_ids 空值", ib.load_ids("") == [])
    check("load_ids 剔除空白项", ib.load_ids('@["a","", " b "]') == ["a", "b"])

    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "new_dir"      # 不存在，应自动创建
        paths, names, outdir = ib.save_batch("http://x", ["id1", "id2", "id3"],
                                             str(out), "shot", 1, 3, "png")
        check("目录自动创建", out.is_dir(), str(outdir))
        check("文件名补零 3 位",
              ib.loads(names) == ["shot_001.png", "shot_002.png", "shot_003.png"],
              str(ib.loads(names)))
        check("落盘文件真实存在",
              all(pathlib.Path(p).is_file() for p in ib.loads(paths)))
        check("落盘内容可解码为 PNG",
              Image.open(io.BytesIO(pathlib.Path(ib.loads(paths)[0]).read_bytes())).size
              == (32, 24))
        # start_index / zero_pad=0
        _, names2, _ = ib.save_batch("http://x", ["id1", "id2"], str(out), "s2", 7, 0, "jpg")
        check("start_index=7 + zero_pad=0",
              ib.loads(names2) == ["s2_7.jpg", "s2_8.jpg"], str(ib.loads(names2)))
        # 空列表：不建目录、不报错
        d2 = pathlib.Path(td) / "empty"
        _, names3, _ = ib.save_batch("http://x", [], str(d2), "e", 1, 3, "png")
        check("空列表 → 空结果且不建目录",
              ib.loads(names3) == [] and not d2.exists())
        try:
            ib.save_batch("http://x", ["id1"], str(out), "x", 1, 3, "tiff")
            check("非法扩展名必须报错", False, "未报错")
        except ValueError as e:
            check("非法扩展名报错", "tiff" in str(e))
        # 取数失败要往上抛（不静默写半截文件）
        def _boom(*a, **k):
            raise RuntimeError("隧道断")
        ib.get_bytes = _boom
        try:
            ib.save_batch("http://x", ["id1"], str(out), "boom", 1, 3, "png")
            check("取图失败必须抛错", False, "未报错")
        except RuntimeError as e:
            check("取图失败抛错", "隧道断" in str(e))
        check("失败时不留半截文件", not (out / "boom_001.png").exists())

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
