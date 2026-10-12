#!/usr/bin/env python3
"""image-batch-load 本机契约单测（不依赖 GPU/推理服务）。

重点：目录扫描口径（pattern 多组 / 排序 / 分页 / 递归）、`@` 信封、
以及"上传失败不许静默"（image_batch.load_batch 会核对返回的 id）。
用法：python3 nodes/image-batch-load/test_main.py
"""
import importlib.util
import pathlib
import sys
import tempfile
import types as _types

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
    # 先把 flowx_client 换成桩（image_batch 只用到 get_bytes/post_bytes）
    fc = _types.ModuleType("flowx_client")
    fc.get_bytes = lambda *a, **k: b"x"
    fc.post_bytes = lambda *a, **k: {"id": "obj"}
    sys.modules["flowx_client"] = fc
    sys.path.insert(0, str(HERE))
    ib = load_mod("image_batch", HERE / "image_batch.py")

    check("dumps/loads 信封往返", ib.loads(ib.dumps(["a", "b"])) == ["a", "b"])
    check("信封前缀是 @", ib.dumps([]).startswith("@"))
    check("容忍裸 JSON", ib.loads('["x"]') == ["x"])
    check("容忍逗号分隔", ib.loads("a, b ,c") == ["a", "b", "c"])
    check("空值 → 空列表", ib.loads("") == [] and ib.loads(None) == [])
    check("pbool 陷阱（bool('false') 为真）",
          ib.pbool("false") is False and ib.pbool("true") is True
          and ib.pbool("0") is False and ib.pbool("1") is True)

    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        for n in ("b.png", "a.png", "c.jpg", "note.txt"):
            (d / n).write_bytes(b"x" * (1 if n == "a.png" else 10))
        sub = d / "sub"
        sub.mkdir()
        (sub / "d.png").write_bytes(b"x" * 5)
        # 排序 + pattern
        files, path = ib.collect(td, "*.png", sort="name")
        check("只匹配 pattern（排除 .jpg/.txt）",
              [pathlib.Path(f).name for f in files] == ["a.png", "b.png"],
              str([pathlib.Path(f).name for f in files]))
        check("返回展开后的绝对目录", path == str(d.resolve()), path)
        files, _ = ib.collect(td, "*.png;*.jpg", sort="name")
        check("pattern 多组（; 分隔）", len(files) == 3, str(len(files)))
        files, _ = ib.collect(td, "*.png;*.jpg", sort="size")
        check("按大小排序（a.png 1B 在前）",
              pathlib.Path(files[0]).name == "a.png", str(files[0]))
        files, _ = ib.collect(td, "*.png", sort="size", reverse="true")
        check("reverse 生效", pathlib.Path(files[0]).name == "b.png", str(files[0]))
        files, _ = ib.collect(td, "*.png", limit=1, offset=1, sort="name")
        check("分页（offset=1 limit=1 → b.png）",
              [pathlib.Path(f).name for f in files] == ["b.png"],
              str([pathlib.Path(f).name for f in files]))
        files, _ = ib.collect(td, "*.png", recursive="true", sort="name")
        check("递归含子目录", len(files) == 3,
              str([pathlib.Path(f).name for f in files]))
        files, _ = ib.collect(td, "*.png", limit=0, offset=99)
        check("offset 超出长度 → 空", files == [])
        try:
            ib.collect("/no/such/dir/xyz", "*.png")
            check("目录不存在必须报错", False, "未报错")
        except FileNotFoundError as e:
            check("目录不存在报错带路径", "xyz" in str(e))
        try:
            ib.collect(td, "*", sort="bogus")
            check("sort 非法必须报错", False, "未报错")
        except ValueError:
            check("sort 非法报错", True)
        check("content_type 映射", ib.content_type("/a/b.PNG") == "image/png"
              and ib.content_type("/a/b.jpg") == "image/jpeg")
        # 上传：桩返回 id ⇒ 成功；返回空 ⇒ 必须报错（不许静默）
        ids, names = ib.load_batch("http://x", [str(d / "a.png"), str(d / "b.png")])
        check("上传产出 ID 与文件名列表",
              ib.loads(ids) == ["obj", "obj"] and ib.loads(names) == ["a.png", "b.png"])
        # ⚠️ 必须改**模块里的名字**：`from flowx_client import post_bytes` 在导入时
        # 就绑定了函数对象，改 sys.modules 里那个模块属性对它无效
        ib.post_bytes = lambda *a, **k: {}
        try:
            ib.load_batch("http://x", [str(d / "a.png")])
            check("上传无 id 必须报错（不静默）", False, "未报错")
        except RuntimeError as e:
            check("上传无 id 报错", "a.png" in str(e))

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
