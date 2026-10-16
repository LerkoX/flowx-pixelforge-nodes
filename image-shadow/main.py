"""image-shadow：环境投影 / 商品落地阴影（G11 电商链路补齐）。

对标电商主图与场景图的「落地阴影」：把主体蒙版按 `distance`/`angle` 位移、以**落地线**
（主体包围盒底边）为锚垂直压扁 `squash`、高斯模糊 `blur`、颜色化并按 `opacity` 上透明
（`fade` 让远端更淡），再按 `canvas` 模式合成回底图或输出带 alpha 的 RGBA。

三条硬契约（与 image-watermark 同口径）：
- **投影区之外逐字节不变**：合成走 PIL `paste(..., mask=...)` 逐像素整数混合，mask=0 处
  一个字节都不动；`shadow_box` = 投影 alpha>0 的包围盒（**不是**近似框），复核脚本裁掉该
  区域后可与输入逐字节比对。
- **确定性**：算法无随机源，同参两跑产物 sha256 相同。
- 纯 PIL、**零 numpy**（手机本地执行器能跑；docker 执行器同样可用）。
"""
import json
import math
import time

from PIL import Image, ImageFilter

from flowx_client import emit_input_media, emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg

CANVAS_MODES = ("expand", "keep")


def parse_color(s, default=(0, 0, 0)):
    """'#RRGGBB' / '#RRGGBBAA' / 'r,g,b' → (r,g,b)。"""
    t = str(s or "").strip()
    if not t:
        return default
    if t.startswith("#"):
        h = t[1:]
        if len(h) in (6, 8):
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        raise ValueError(f"image-shadow: 颜色非法 {s!r}（用 #RRGGBB 或 #RRGGBBAA）")
    parts = [p.strip() for p in t.split(",")]
    if len(parts) != 3:
        raise ValueError(f"image-shadow: 颜色非法 {s!r}（用 #RRGGBB 或 r,g,b）")
    return tuple(max(0, min(255, int(float(p)))) for p in parts)


def subject_alpha(image, mask=None):
    """主体不透明度（L 图）：mask 优先（灰度语义 255=主体），否则取 image 的 alpha；
    纯 RGB 无蒙版时视为「整图都是主体」。"""
    if mask is not None:
        m = mask.convert("L")
        if m.size != image.size:
            m = m.resize(image.size, Image.LANCZOS)
        return m
    if image.mode in ("RGBA", "LA"):
        return image.convert("RGBA").getchannel("A")
    return Image.new("L", image.size, 255)


def content_bbox(alpha, threshold=1):
    """内容包围盒（>=threshold 视为有内容）；空蒙版返回 None。"""
    if alpha is None:
        return None
    return alpha.point(lambda v: 255 if v >= threshold else 0).getbbox()


def paste_clamped(dst, src, xy):
    """把 src 贴到 dst 的 xy 处，**dst 未被覆盖的区域用 src 的边缘像素延展**（不留白条）。

    为什么需要：canvas=expand 时画布被影子撑大，而背景图未必盖得住新画布，直接 paste
    会在边缘留一条白色（实测 exec 716 左侧 58px 白条）——那显然不是商品图的语义。
    边缘延展是确定性、无重采样的做法（同 image-pad 的 fill=edge）。
    """
    x, y = int(xy[0]), int(xy[1])
    w, h = src.size
    W, H = dst.size
    dst.paste(src, (x, y))  # 中间区域（越界部分 PIL 自动裁掉）
    if x > 0:  # 左条：src 第 0 列横向延展
        dst.paste(src.crop((0, 0, 1, h)).resize((x, h), Image.NEAREST), (0, y))
    if x + w < W:  # 右条
        dst.paste(src.crop((w - 1, 0, w, h)).resize((W - x - w, h), Image.NEAREST),
                  (x + w, y))
    if y > 0:  # 上条
        dst.paste(src.crop((0, 0, w, 1)).resize((w, y), Image.NEAREST), (x, 0))
    if y + h < H:  # 下条
        dst.paste(src.crop((0, h - 1, w, h)).resize((w, H - y - h), Image.NEAREST),
                  (x, y + h))
    # 四个角：用对应角的 1px 像素铺满
    if x > 0 and y > 0:
        dst.paste(src.crop((0, 0, 1, 1)).resize((x, y), Image.NEAREST), (0, 0))
    if x + w < W and y > 0:
        dst.paste(src.crop((w - 1, 0, w, 1)).resize((W - x - w, y), Image.NEAREST),
                  (x + w, 0))
    if x > 0 and y + h < H:
        dst.paste(src.crop((0, h - 1, 1, h)).resize((x, H - y - h), Image.NEAREST),
                  (0, y + h))
    if x + w < W and y + h < H:
        dst.paste(src.crop((w - 1, h - 1, w, h))
                  .resize((W - x - w, H - y - h), Image.NEAREST),
                  (x + w, y + h))
    return dst


def shadow_offset(distance, angle):
    """位移向量（整数像素；取整 ⇒ 无重采样不确定性）。"""
    return (int(round(float(distance) * math.cos(math.radians(float(angle))))),
            int(round(float(distance) * math.sin(math.radians(float(angle))))))


def build_shadow(alpha, distance=24, angle=90.0, blur=24.0, opacity=0.45,
                 squash=0.55, fade=0.0, color=(0, 0, 0)):
    """在**给定蒙版同尺寸**的画布上构建投影层。

    返回 (shadow_rgba, shadow_alpha, offset(dx,dy), peak_alpha)：调用方负责给足画布余量
    （越界部分会被裁掉，这是 canvas=keep 的语义；expand 由调用方先加 pad）。
    """
    bbox = content_bbox(alpha)
    if bbox is None:
        return None, None, (0, 0), 0
    w, h = alpha.size
    ground = bbox[3]  # 落地线 = 主体包围盒底边

    # 1) 压扁：只压缩主体高度区间，底部对齐落地线（贴地透视）
    squashed = alpha
    if float(squash) < 1.0:
        sub = alpha.crop((0, bbox[1], w, ground))
        nh = max(1, int(round(sub.height * float(squash))))
        sub = sub.resize((w, nh), Image.BILINEAR)
        squashed = Image.new("L", (w, h), 0)
        squashed.paste(sub, (0, max(0, ground - nh)))

    # 2) 位移
    dx, dy = shadow_offset(distance, angle)
    moved = squashed
    if dx or dy:
        moved = Image.new("L", (w, h), 0)
        moved.paste(squashed, (dx, dy))

    # 3) 高斯模糊
    if blur and float(blur) > 0:
        moved = moved.filter(ImageFilter.GaussianBlur(float(blur)))

    # 4) 远端淡出：沿位移方向的线性梯度（远端 = 位移指向的一侧，更淡）
    f = max(0.0, min(1.0, float(fade)))
    if f > 0 and (dx or dy):
        # linear_gradient 是「上 0 下 255」；rotate(90-angle) 让 255 落在**位移指向**的一侧
        # （angle=0 右 / 90 下 / 180 左 / 270 上 ⇒ 远端 t=255，乘 (255-255f) 变最淡）。
        # 再把梯度拉伸到**影子自身的范围**（t: 近端 0 → 远端 255）：否则影子只占画布一角时
        # fade 几乎看不出来（梯度按整幅画布算，t 永远用不到 0/255 两端）。
        grad = Image.linear_gradient("L").resize((w, h))
        grad = grad.rotate(90.0 - float(angle), resample=Image.BILINEAR)
        sbb = content_bbox(moved)
        if sbb is not None:
            g = grad.crop(sbb)
            lo, hi = g.getextrema()
            span = max(1, hi - lo)
            lut = [max(0, min(255, int((v - lo) * 255 / span))) for v in range(256)]
            grad = Image.new("L", (w, h), 0)
            grad.paste(g.point(lut), (sbb[0], sbb[1]))
        src_bytes, grad_bytes = moved.tobytes(), grad.tobytes()
        moved = Image.frombytes("L", (w, h), bytes(
            (v * (255 - (g * f).__int__())) // 255
            for v, g in zip(src_bytes, grad_bytes)))

    # 5) 不透明度定标
    scale = int(round(max(0.0, min(1.0, float(opacity))) * 255))
    shadow_alpha = moved.point(lambda v: v * scale // 255)
    peak = shadow_alpha.getextrema()[1] if content_bbox(shadow_alpha) else 0
    r, g, b = color
    shadow_rgba = Image.new("RGBA", (w, h), (r, g, b, 0))
    shadow_rgba.putalpha(shadow_alpha)
    return shadow_rgba, shadow_alpha, (dx, dy), peak


def process(image, mask=None, background=None, distance=24, angle=90.0, blur=24.0,
            opacity=0.45, squash=0.55, fade=0.0, color=(0, 0, 0), canvas="expand"):
    """主流程。返回 (out_image, shadow_box | None, meta)。

    canvas="expand"：画布外扩到刚好容纳（主体 ∪ 投影）；主体像素位置不变、逐字节保留；
    canvas="keep"  ：保持输入尺寸，越界投影裁掉。
    background 给定 ⇒ 输出 RGB（投影在底图之上、主体压在投影之上）；
    未给 ⇒ 输出 RGBA 透明画布（供下游 image-composite）。
    """
    src = image.convert("RGBA")
    w, h = src.size
    alpha = subject_alpha(src, mask)
    dx, dy = shadow_offset(distance, angle)
    # expand 模式：先在四边留足余量（位移 + 高斯尾巴），避免投影被画布边界削掉
    pad = 0
    if canvas == "expand":
        pad = int(abs(dx) + abs(dy) + max(0.0, float(blur)) * 2) + 2
    W, H = w + 2 * pad, h + 2 * pad

    big_alpha = Image.new("L", (W, H), 0)
    big_alpha.paste(alpha, (pad, pad))
    shadow_rgba, shadow_alpha, (dx, dy), peak = build_shadow(
        big_alpha, distance=distance, angle=angle, blur=blur, opacity=opacity,
        squash=squash, fade=fade, color=color)

    # 最终画布：expand = 主体 ∪ 投影 的包围盒；keep = 主体整幅
    sbox = content_bbox(shadow_alpha)
    subj = (pad, pad, pad + w, pad + h)
    if canvas == "expand" and sbox is not None:
        x0 = min(sbox[0], subj[0])
        y0 = min(sbox[1], subj[1])
        x1 = max(sbox[2], subj[2])
        y1 = max(sbox[3], subj[3])
    else:
        x0, y0, x1, y1 = subj
    box = (x0, y0, x1, y1)
    ox, oy = x0, y0

    if background is not None:
        bg = background.convert("RGB")
        out = Image.new("RGB", (x1 - x0, y1 - y0), (255, 255, 255))
        # 底图按原尺寸贴在主体位置；画布被 expand 撑大后盖不满的部分做边缘延展（不留白条）
        paste_clamped(out, bg, (pad - x0, pad - y0))
        if shadow_alpha is not None:
            out.paste(shadow_rgba.crop(box).convert("RGB"), (0, 0),
                      shadow_alpha.crop(box))
        # 主体压在投影之上（mask 混合；src == 底图处逐字节不变）
        out.paste(src.convert("RGB"), (pad - x0, pad - y0), alpha)
    else:
        # ⚠️ 透明画布必须用 alpha_composite（source-over）：paste(…, mask=自身 alpha)
        # 会把 alpha 平方（dst=0 时得到 a²/(255·r)），远端半透明边缘会失真
        out = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        if shadow_alpha is not None:
            out.alpha_composite(shadow_rgba.crop(box))
        out.alpha_composite(src, dest=(pad - x0, pad - y0))

    # 投影框换算到输出画布坐标
    out_box = (sbox[0] - ox, sbox[1] - oy, sbox[2] - ox, sbox[3] - oy) \
        if sbox is not None else None
    return out, out_box, {"offset": [dx, dy], "peak_alpha": peak,
                          "input_size": [w, h],
                          # 主体在输出画布里的左上角（expand 时非 0；复核脚本据此定位原图区）
                          "subject_at": [pad - ox, pad - oy],
                          "canvas_size": [out.width, out.height]}


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    # 输入原图物化：画布对比控件的「原图」侧（推理服务对象 GC/TTL 后
    # 必然 404，必须在执行当时落一份到 Studio 媒体缓存）
    emit_input_media(url, param("image"), tok)
    # 先取参数（gate N 桩冒烟要求 main() 里所有名字都有定义）
    image_id = param("image")
    mask_id = param("mask", "")
    bg_id = param("background", "")
    distance = param("distance", 24, cast=int)
    angle = param("angle", 90, cast=float)
    blur = param("blur", 24, cast=float)
    opacity = param("opacity", 0.45, cast=float)
    squash = param("squash", 0.55, cast=float)
    fade = param("fade", 0.0, cast=float)
    color = parse_color(param("color", "#000000"))
    canvas = param("canvas", "expand")
    if canvas not in CANVAS_MODES:
        raise ValueError(f"image-shadow: canvas 非法 {canvas!r}（expand|keep）")
    t0 = time.time()

    im = get_pimg(url, image_id, tok, mode="RGBA")
    mask = get_pimg(url, mask_id, tok, mode="L") if mask_id else None
    bg = get_pimg(url, bg_id, tok, mode="RGB") if bg_id else None

    out, sbox, meta = process(im, mask=mask, background=bg, distance=distance,
                              angle=angle, blur=blur, opacity=opacity,
                              squash=squash, fade=fade, color=color, canvas=canvas)
    oid = post_pimg(url, out, tok)
    cov = 0.0
    if sbox is not None:
        cov = round((sbox[2] - sbox[0]) * (sbox[3] - sbox[1]) /
                    float(out.width * out.height), 4)
    box = ",".join(str(v) for v in sbox) if sbox else ""
    info = json.dumps({
        "distance": distance, "angle": angle, "blur": blur, "opacity": opacity,
        "squash": squash, "fade": fade, "canvas": canvas,
        # ⚠️ 颜色在这里用 "r,g,b" 写，**不能带 #**：Studio 的 codec-block 提取器
        # 用 `\s*#.*$` 无差别删行尾注释 ⇒ 引号里的 #RRGGBB 也会被截断 ⇒ 整块 YAML
        # 解析失败、该节点**所有输出丢失**（exec 714 实测踩到；parse_color 两种都认）。
        "color": "%d,%d,%d" % color, "offset": meta["offset"],
        "peak_alpha": meta["peak_alpha"], "input_size": meta["input_size"],
        "subject_at": meta["subject_at"],
        "canvas_size": [out.width, out.height], "shadow_box": box,
        "shadow_coverage": cov,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }, ensure_ascii=False)
    print(f"[image-shadow] in={im.size} shadow={box or '-'} cov={cov:.1%} "
          f"peak_alpha={meta['peak_alpha']} canvas={canvas} -> {oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height, shadow_box=box,
         coverage=cov, info=info)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
