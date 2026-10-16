from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("seg.sam_auto", {
        "image": ref(param("image")),
        "model": param("model", "sam:vit_l"),
        "points_per_side": int(param("points_per_side", "16")),
        "pred_iou_thresh": float(param("pred_iou_thresh", "0.88")),
        "stability_score_thresh": float(param("stability_score_thresh", "0.95")),
        "min_area": int(param("min_area", "200")),
        "max_masks": int(param("max_masks", "64")),
        "keep": param("keep", ""),
        "sort": param("sort", "area"),
        "overlay": float(param("overlay", "0.45")),
    }, emit_keys=["masks", "selected", "selected_overlay", "mask_ids", "count",
                  "kept", "sizes", "montage_cols", "montage_tile", "union_area", "bbox"], timeout=1800, input_media=("image",))


if __name__ == "__main__":
    main()
