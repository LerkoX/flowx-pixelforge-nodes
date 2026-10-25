"""image-compare：原图 vs 结果图对比展示（纯展示节点，不做图像处理）。

- emit_input_media：把「原图」物化到 Studio 媒体缓存（kind=input）——推理服务
  对象是内存态，GC/重启/TTL 后必 404，物化产物随媒体缓存持久可读，是画布
  对比控件「原图」侧的首选来源。
- emit_result_preview：把「结果图」上报为最终预览帧（progress=1.0），即画布
  对比控件「结果」侧（preview 最终帧）。
- outputs 附带 __inputs_resolved（image_before 对象 ID），作物化缺失时画布
  经节点级代理实时拉取的降级链。
"""
import json

from flowx_client import emit, emit_input_media, emit_result_preview, param, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    before = param("image_before")
    after = param("image_after")
    if not before or not after:
        raise RuntimeError("image-compare 需要 image_before 与 image_after 两个图像对象 ID")

    emit_input_media(url, before, tok)
    emit_result_preview(url, after, tok)
    print(f"[image-compare] before={before} after={after}", flush=True)
    emit(before=before, after=after,
         __inputs_resolved=json.dumps({"image_before": before, "image_after": after},
                                      ensure_ascii=False))


if __name__ == "__main__":
    main()
