"""breakout-compose：破框九宫格合成（纯节点侧 PIL，不占显存）。

人物 RGBA 图层按「身高比例 + 中心 x + 脚底 y」锚定叠加到九宫格底图上：
脚底钉在底图下部、身高放大到接近/超过底图高度时，头肩自然越出网格顶边，
形成破框出屏感；可选软投影（距离/方向/模糊/不透明度/颜色）强化
「人物悬浮在网格之上」的 3D 纵深。上游：grid-compose + matting-birefnet。
"""
import json
import math

from flowx_client import emit_input_media, emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg, pbool


def _parse_color(s, default="#000000"):
    """#RGB / #RRGGBB → (r,g,b)。"""
    s = (s or default).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"shadow_color 非法：{s!r}（用 #RRGGBB）")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _solve8(a, b):
    """8x8 高斯消元（纯 Python，避免为透视系数引入 numpy）。"""
    n = 8
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("透视系数奇异（控制点共线？）")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            k = m[r][col] / m[col][col]
            m[r] = [v - k * m[col][j] for j, v in enumerate(m[r])]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][j] * x[j] for j in range(r + 1, n))) / m[r][r]
    return x


def _persp_coeffs(out_quad, in_quad):
    """PIL PERSPECTIVE 系数：out_quad=输出四边形（左右序：左上/右上/右下/左下），in_quad=对应源点。"""
    a, b = [], []
    for (x, y), (ix, iy) in zip(out_quad, in_quad):
        a.append([x, y, 1, 0, 0, 0, -ix * x, -ix * y]); b.append(ix)
        a.append([0, 0, 0, x, y, 1, -iy * x, -iy * y]); b.append(iy)
    return _solve8(a, b)


def tilt_grid(grid, tilt=0.0, tilt_bg="#FFFFFF"):
    """底图俯视透视：顶边两侧各内收 tilt% 宽（上边短下边长=网格向远处倾斜），
    人物保持正直 ⇒ 纵深差强化出框感。tilt=0 原样返回。"""
    from PIL import Image
    t = float(tilt)
    if t <= 0:
        return grid
    t = min(t, 40.0)
    w, h = grid.size
    inset = w * t / 100.0
    src = grid.convert("RGB")
    # 输出四边形（梯形）→ 源矩形四角
    out_quad = [(inset, 0), (w - inset, 0), (w, h), (0, h)]
    in_quad = [(0, 0), (w, 0), (w, h), (0, h)]
    coeffs = _persp_coeffs(out_quad, in_quad)
    return src.transform(src.size, Image.PERSPECTIVE, coeffs,
                         resample=Image.BICUBIC, fillcolor=_parse_color(tilt_bg, "#FFFFFF"))


def _composite_clipped(canvas, layer, x0, y0):
    """把 layer（RGBA）alpha 合成到 canvas（RGBA）的 (x0,y0)，越界部分裁掉。"""
    from PIL import Image
    cw, ch = canvas.size
    lw, lh = layer.size
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(cw, x0 + lw), min(ch, y0 + lh)
    if ix1 <= ix0 or iy1 <= iy0:
        return
    region = layer.crop((ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
    base = canvas.crop((ix0, iy0, ix1, iy1))
    canvas.paste(Image.alpha_composite(base, region), (ix0, iy0))


def person_box(grid_size, person_size, height_ratio=0.9, center_x=50.0, feet_y=95.0):
    """计算人物缩放后尺寸与左上角落点（可越界，负值=破框）。"""
    gw, gh = grid_size
    pw, ph = person_size
    th = max(1, round(gh * float(height_ratio)))
    tw = max(1, round(pw * th / ph))
    x0 = round(float(center_x) / 100.0 * gw - tw / 2)
    y0 = round(float(feet_y) / 100.0 * gh - th)
    return x0, y0, tw, th


def trim_alpha(person, pad=0):
    """把 RGBA 人物裁到 alpha 包围盒（去透明留白），height_ratio 才是真实身高占比。"""
    bbox = person.getchannel("A").getbbox()
    if not bbox:
        raise ValueError("人物图层 alpha 全空（不是有效的去背图）")
    if pad > 0:
        l, t, r, b = bbox
        bbox = (max(0, l - pad), max(0, t - pad),
                min(person.width, r + pad), min(person.height, b + pad))
    return person.crop(bbox)


def compose(grid, person, height_ratio=0.9, center_x=50.0, feet_y=95.0, flip=False,
            shadow=True, shadow_distance=30, shadow_angle=90.0, shadow_blur=30.0,
            shadow_opacity=0.4, shadow_color="#000000", trim=True,
            tilt=0.0, tilt_bg="#FFFFFF"):
    """合成破框海报；返回 (RGB 图, person_box)。"""
    from PIL import Image, ImageFilter
    grid = tilt_grid(grid, tilt, tilt_bg)
    person = person.convert("RGBA")
    if trim:
        person = trim_alpha(person)
    x0, y0, tw, th = person_box(grid.size, person.size, height_ratio, center_x, feet_y)
    p = person.resize((tw, th), Image.LANCZOS)
    if flip:
        p = p.transpose(Image.FLIP_LEFT_RIGHT)

    canvas = grid.convert("RGBA")
    if shadow and float(shadow_opacity) > 0:
        rad = math.radians(float(shadow_angle))
        dx = round(float(shadow_distance) * math.cos(rad))
        dy = round(float(shadow_distance) * math.sin(rad))
        # 画布尺寸透明层上放人物剪影 → 整体高斯模糊（无边缘截断伪影）
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sil = Image.new("RGBA", p.size, _parse_color(shadow_color) + (255,))
        sil.putalpha(p.getchannel("A"))
        _composite_clipped(layer, sil, x0 + dx, y0 + dy)
        if float(shadow_blur) > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(float(shadow_blur)))
        r, g, b, a = layer.split()
        a = a.point(lambda v: int(v * float(shadow_opacity)))
        canvas.alpha_composite(Image.merge("RGBA", (r, g, b, a)))

    _composite_clipped(canvas, p, x0, y0)
    return canvas.convert("RGB"), {"x": x0, "y": y0, "width": tw, "height": th}


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    grid_id = param("grid")
    person_id = param("person")
    # 输入原图物化：画布对比控件的「原图」侧用九宫格底图
    emit_input_media(url, grid_id, tok)
    grid = get_pimg(url, grid_id, tok, mode="RGB")
    person = get_pimg(url, person_id, tok, mode="RGBA")
    out, box = compose(grid, person,
                       height_ratio=param("height_ratio", 0.9, cast=float),
                       center_x=param("center_x", 50.0, cast=float),
                       feet_y=param("feet_y", 95.0, cast=float),
                       flip=pbool(param("flip", "false")),
                       shadow=pbool(param("shadow", "true")),
                       shadow_distance=param("shadow_distance", 30, cast=int),
                       shadow_angle=param("shadow_angle", 90.0, cast=float),
                       shadow_blur=param("shadow_blur", 30.0, cast=float),
                       shadow_opacity=param("shadow_opacity", 0.4, cast=float),
                       shadow_color=param("shadow_color", "#000000"),
                       trim=pbool(param("trim", "true")),
                       tilt=param("tilt", 0.0, cast=float),
                       tilt_bg=param("tilt_bg", "#FFFFFF"))
    oid = post_pimg(url, out, tok)
    print(f"[breakout-compose] grid={grid_id} person={person_id} box={box} -> image={oid}",
          flush=True)
    emit(image=oid, width=out.width, height=out.height, person_box=json.dumps(box))
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
