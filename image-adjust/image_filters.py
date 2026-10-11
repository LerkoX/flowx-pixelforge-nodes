"""image_filters.py：传统图像后处理共享实现（G5 滤镜族，纯 PIL）。

对标 ComfyUI-Image-Filters / WAS Node Suite 的常用滤镜，做成细粒度节点：
adjust / hsl / curves / blur / sharpen / noise / channel-extract / channel-merge。

设计约束：
- **纯 PIL**（节点侧执行，无 numpy / opencv 依赖 —— 节点镜像只装了 Pillow）；
- 所有函数签名 `f(im, **params) -> PIL.Image`，参数已由节点侧 cast 成正确类型；
- 保持输入尺寸与模式（RGB 进 RGB 出；channel-extract 故意输出 L）；
- 亮度/对比度/伽马/锐度、饱和度的 1.0 与 hue/lightness 的 0 均为"不变"；
- 逐像素操作一律走 `point(list)`（C 级 LUT），不用 Python lambda 回调。

单一事实源：本文件在 `nodes/_common/`，由 `_tools/sync-common.py` 分发到各成员
节点目录（`check-bundle.py` 校验副本一致，禁止直接改节点目录里的副本）。
"""
import math
import random

from PIL import Image, ImageChops, ImageEnhance, ImageFilter


def _bool(raw):
    """env 注入的布尔参数（'false' 字符串为真，须显式解析）。"""
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- adjust


def _gamma_lut(gamma):
    """gamma LUT：gamma>1 变亮（社区通用语义）。"""
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma!r}")
    return [min(255, int(255.0 * ((i / 255.0) ** (1.0 / gamma)) + 0.5))
            for i in range(256)]


def adjust(im, brightness=1.0, contrast=1.0, gamma=1.0, sharpness=1.0):
    """亮度 / 对比度 / 伽马 / 锐度（1.0 = 不变）。"""
    out = im
    if float(brightness) != 1.0:
        out = ImageEnhance.Brightness(out).enhance(float(brightness))
    if float(contrast) != 1.0:
        out = ImageEnhance.Contrast(out).enhance(float(contrast))
    if float(gamma) != 1.0:
        lut = _gamma_lut(float(gamma))
        out = out.point(lut * len(out.getbands()))
    if float(sharpness) != 1.0:
        out = ImageEnhance.Sharpness(out).enhance(float(sharpness))
    return out


# ---------------------------------------------------------------- hsl


def hsl(im, hue=0.0, saturation=1.0, lightness=0.0):
    """色相偏移（-180~180 度）/ 饱和度倍率 / 明度加性偏移（-1~1，×255）。

    中性参数（0/1/0）直接返回原图：RGB→HSV→RGB 往返有 8bit 量化损失，
    no-op 必须做到逐像素幂等。
    """
    if float(hue) == 0.0 and float(saturation) == 1.0 and float(lightness) == 0.0:
        return im
    hsv = im.convert("RGB").convert("HSV")
    h, s, v = hsv.split()
    shift = int(round(float(hue) / 360.0 * 256.0)) % 256
    if shift:
        h = h.point([(i + shift) % 256 for i in range(256)])
    sat = float(saturation)
    if sat != 1.0:
        s = s.point([min(255, max(0, int(i * sat + 0.5))) for i in range(256)])
    off = int(round(float(lightness) * 255))
    if off:
        v = v.point([min(255, max(0, i + off)) for i in range(256)])
    return Image.merge("HSV", (h, s, v)).convert("RGB")


# ---------------------------------------------------------------- curves


def parse_curve_points(raw, default=((0, 0), (255, 255))):
    """解析 `"x,y;x,y;..."` 控制点串 → 按 x 升序的点列表。

    容错：空/坏串 → default；坐标 clamp 0~255；重复 x 保留后者；非数字项跳过。
    """
    if raw is None:
        return list(default)
    if isinstance(raw, (list, tuple)) and raw and not isinstance(raw[0], str):
        items, as_text = list(raw), False
    else:
        text = str(raw).strip().strip(";")
        if not text:
            return list(default)
        items, as_text = text.split(";"), True
    pts = []
    for item in items:
        try:
            if as_text:
                xs, ys = str(item).split(",")
                x, y = float(xs), float(ys)
            else:
                x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        pts.append((max(0.0, min(255.0, x)), max(0.0, min(255.0, y))))
    if not pts:
        return list(default)
    pts.sort(key=lambda p: p[0])
    dedup = []
    for p in pts:
        if dedup and p[0] == dedup[-1][0]:
            dedup[-1] = p
        else:
            dedup.append(p)
    return dedup


def curve_lut(points):
    """控制点 → 256 项 LUT（相邻点分段线性插值，两端以首/末点值延伸）。"""
    pts = list(points) or [(0, 0), (255, 255)]
    lut = []
    for i in range(256):
        if i <= pts[0][0]:
            lut.append(int(round(pts[0][1])))
        elif i >= pts[-1][0]:
            lut.append(int(round(pts[-1][1])))
        else:
            val = i
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                if x0 <= i <= x1:
                    val = y1 if x1 == x0 else y0 + (i - x0) / (x1 - x0) * (y1 - y0)
                    break
            lut.append(int(round(val)))
    return [min(255, max(0, v)) for v in lut]


def curves(im, channel="all", points="0,0;255,255"):
    """曲线映射：channel=all 作用于所有 band，否则只作用于指定单通道。"""
    lut = curve_lut(parse_curve_points(points))
    ch = str(channel or "all").strip().lower()
    rgb = im.convert("RGB")
    if ch in ("all", "rgb", ""):
        return rgb.point(lut * 3)
    names = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2}
    if ch not in names:
        raise ValueError(f"channel must be all|red|green|blue, got {channel!r}")
    bands = list(rgb.split())
    idx = names[ch]
    bands[idx] = bands[idx].point(lut)
    return Image.merge("RGB", bands)


# ---------------------------------------------------------------- blur


def _motion_kernel(radius, angle):
    """方向运动模糊核（PIL `ImageFilter.Kernel` 只支持 3x3/5x5 → 固定 5x5）。

    任意半径由 `blur()` 重复卷积实现（线段核自卷积 = 更长的线段核）。
    """
    radius = max(1, min(2, int(radius)))
    size = radius * 2 + 1
    rad = math.radians(float(angle))
    dx, dy = math.cos(rad), math.sin(rad)
    rows, hits = [], 0
    for j in range(size):
        row = []
        for i in range(size):
            px, py = i - radius, j - radius
            on = (abs(px * dy - py * dx) <= 0.5
                  and abs(px * dx + py * dy) <= radius + 0.5)
            row.append(1.0 if on else 0.0)
            hits += 1 if on else 0
        rows.append(row)
    if hits == 0:  # 退化保护 → 水平线
        rows = [[1.0 if j == radius else 0.0 for j in range(size)]
                for _ in range(size)]
        hits = 1
    return size, [v / hits for row in rows for v in row]


def blur(im, mode="gaussian", radius=4.0, angle=0.0):
    """高斯 / box / 方向运动模糊。"""
    m = str(mode or "gaussian").strip().lower()
    r = float(radius)
    if r <= 0:
        return im
    if m == "gaussian":
        return im.filter(ImageFilter.GaussianBlur(r))
    if m == "box":
        return im.filter(ImageFilter.BoxBlur(r))
    if m in ("motion", "directional"):
        # 5x5 线段核重复卷积 ≈ 长度为 2*radius+1 的运动拖影（C 级、无 numpy）
        size, kernel = _motion_kernel(2, angle)
        passes = max(1, int(round(r / 2.0)))
        out = im
        for _ in range(passes):
            out = out.filter(ImageFilter.Kernel((size, size), kernel, scale=1.0))
        return out
    raise ValueError(f"mode must be gaussian|box|motion, got {mode!r}")


# ---------------------------------------------------------------- sharpen


def sharpen(im, mode="usm", radius=2.0, percent=150.0, threshold=3.0):
    """USM 锐化（radius/percent/threshold）或简单锐化（percent 当强度倍率）。"""
    m = str(mode or "usm").strip().lower()
    if m in ("usm", "unsharp"):
        return im.filter(ImageFilter.UnsharpMask(
            radius=float(radius), percent=int(percent), threshold=int(threshold)))
    if m == "simple":
        return ImageEnhance.Sharpness(im).enhance(float(percent) / 100.0)
    raise ValueError(f"mode must be usm|simple, got {mode!r}")


# ---------------------------------------------------------------- noise


def _rand_l(seed, width, height):
    """seeded 均匀噪声灰度图（randbytes 为 C 级，快且完全可复现）。"""
    rng = random.Random(int(seed))
    return Image.frombytes("L", (width, height), rng.randbytes(width * height))


def _gauss_layers(width, height, sigma, count):
    """高斯噪声层（PIL effect_noise，C 实现；每层独立噪声）。"""
    return [Image.effect_noise((width, height), sigma) for _ in range(count)]


def noise(im, mode="gaussian", amount=0.05, monochrome=False, seed=0):
    """噪点叠加：gaussian / uniform / salt_pepper，amount 0~1。

    可复现性：uniform / salt_pepper 完全由 seed 决定；gaussian 走 PIL
    `Image.effect_noise`（C 实现，跨 Pillow 版本不保证逐像素一致）。
    """
    amt = float(amount)
    if amt <= 0:
        return im
    base = im.convert("RGB")
    w, h = base.size
    m = str(mode or "gaussian").strip().lower()
    mono = _bool(monochrome)
    n_ch = 1 if mono else 3

    if m in ("gaussian", "gauss"):
        sigma = max(1.0, amt * 128.0)
        layers = _gauss_layers(w, h, sigma, n_ch)
        noise_rgb = Image.merge("RGB", layers if not mono else layers * 3)
        # noise 中心在 128 → offset -128 使其成为零均值增量
        return ImageChops.add(base, noise_rgb, scale=1.0, offset=-128)

    if m in ("uniform", "u"):
        span = max(1.0, amt * 255.0)
        layers = [_rand_l(int(seed) + i, w, h) for i in range(n_ch)]
        noise_rgb = Image.merge("RGB", layers if not mono else layers * 3)
        # 0~255 均匀值 → 128 ± span（之后 offset -128 得 ±span 增量）
        lut = [min(255, max(0, int(round(128 + (i / 255.0 * 2 - 1) * span))))
               for i in range(256)]
        return ImageChops.add(noise_rgb.point(lut * 3), base, scale=1.0, offset=-128)

    if m in ("salt_pepper", "saltpepper", "sp"):
        p = max(0.0, min(1.0, amt)) / 2.0
        white_at = max(1, int(round((1.0 - p) * 255)))
        black_at = int(round(p * 255))
        lut_kind = [255 if i >= white_at else (0 if i <= black_at else 128)
                    for i in range(256)]
        lut_white = [255 if v == 255 else 0 for v in range(256)]
        lut_black = [255 if v == 0 else 0 for v in range(256)]
        bands = list(base.split())
        for c in range(3):
            kind = _rand_l(int(seed) + (0 if mono else c), w, h).point(lut_kind)
            bands[c] = Image.composite(Image.new("L", (w, h), 255), bands[c],
                                       kind.point(lut_white))
            bands[c] = Image.composite(Image.new("L", (w, h), 0), bands[c],
                                       kind.point(lut_black))
        return Image.merge("RGB", bands)

    raise ValueError(f"mode must be gaussian|uniform|salt_pepper, got {mode!r}")


# ---------------------------------------------------------------- channels


def channel_extract(im, channel="luminance"):
    """导出单通道灰度图（可直接当蒙版喂 mask-* 节点）。"""
    ch = str(channel or "luminance").strip().lower()
    if ch in ("luminance", "l", "gray", "grey"):
        return im.convert("L")
    idx = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2}
    if ch in idx:
        return im.convert("RGB").split()[idx[ch]]
    if ch in ("alpha", "a"):
        if im.mode in ("RGBA", "LA", "PA"):
            return im.convert("RGBA").split()[3]
        return Image.new("L", im.size, 255)
    raise ValueError(f"channel must be red|green|blue|alpha|luminance, got {channel!r}")


def channel_merge(r_image, g_image, b_image, alpha_image=None):
    """三路灰度图合并为 RGB（给 alpha 时输出 RGBA）。尺寸以 R 路为准。"""
    for name, img in (("r", r_image), ("g", g_image), ("b", b_image)):
        if img is None:
            raise ValueError(f"{name}_image is required")
    size = r_image.size
    bands = []
    for img in (r_image, g_image, b_image):
        conv = img.convert("L")
        if conv.size != size:
            conv = conv.resize(size, Image.Resampling.BILINEAR)
        bands.append(conv)
    if alpha_image is not None:
        a = alpha_image.convert("L")
        if a.size != size:
            a = a.resize(size, Image.Resampling.BILINEAR)
        return Image.merge("RGBA", bands + [a])
    return Image.merge("RGB", bands)
