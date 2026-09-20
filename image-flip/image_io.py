"""图像对象 <-> PIL 互转共享辅助（第一档纯客户端图像/蒙版节点共用）。

mask 约定：内容为灰度的 IMAGE 对象。读入一律 convert("L")（服务端
POST /images 会转 RGB，灰度信息保留在三个相同通道中，取 L 即可还原）。
每个节点包通过 flowx.json 的 files 字段携带本文件。"""
import base64
import io

from PIL import Image

from flowx_client import get_bytes, post_bytes

RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def get_pimg(base, oid, tok=None, mode=None):
    """下载图像对象为 PIL.Image；mode 给定时转换（mask 节点传 "L"）。"""
    data = get_bytes(base, f"/images/{oid}", tok, timeout=300)
    im = Image.open(io.BytesIO(data))
    im.load()
    return im.convert(mode) if mode else im


def post_pimg(base, pil, tok=None):
    """PIL.Image 编码 PNG 上传，返回新图像对象 id。"""
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    resp = post_bytes(base, "/images", buf.getvalue(), tok, timeout=300,
                      content_type="image/png")
    return resp["id"]


def thumb_data_url(pil, max_side=256, quality=75):
    """结果缩略图 data URL（JPEG），经 Metadata 带给画布 widget 预览——
    媒体本体仍留在服务端对象仓库，缩略图仅做画布展示（save-video 旧内嵌路线）。"""
    t = pil.copy().convert("RGB")
    t.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pbool(raw):
    """解析布尔参数（env 注入为字符串，bool("false") 为真，须显式解析）。"""
    return str(raw).strip().lower() in ("1", "true", "yes", "on")
