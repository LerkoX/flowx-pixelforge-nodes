#!/usr/bin/env python3
"""build-widget.py：拼接 _common/widget-base.js + <node>/ui/widget-def.js
生成最终单文件 <node>/ui/node-widget.js（widget 分发要求单文件无依赖）。

用法：python3 _tools/build-widget.py <node-dir> [<node-dir>...]
      python3 _tools/build-widget.py --all   # 重建所有含 widget-def.js 的节点
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "_common" / "widget-base.js"
FOOTER = "\nexport default createNodeWidget(WIDGET_SPEC)\n"


def build(node_dir: pathlib.Path):
    def_file = node_dir / "ui" / "widget-def.js"
    out_file = node_dir / "ui" / "node-widget.js"
    if not def_file.is_file():
        print(f"{node_dir.name}: 无 ui/widget-def.js，跳过", file=sys.stderr)
        return False
    content = (BASE.read_text() + "\n// ---- 节点专属定义（widget-def.js，"
               "由 build-widget.py 拼接，请勿直接编辑本文件）----\n"
               + def_file.read_text() + FOOTER)
    out_file.write_text(content)
    print(f"{node_dir.name}: -> ui/node-widget.js ({len(content)} bytes)")
    return True


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    if args == ["--all"]:
        dirs = sorted(p.parent.parent for p in ROOT.glob("*/ui/widget-def.js"))
    else:
        dirs = [pathlib.Path(a) if pathlib.Path(a).is_absolute()
                else ROOT / a for a in args]
    ok = all(build(d) for d in dirs)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
