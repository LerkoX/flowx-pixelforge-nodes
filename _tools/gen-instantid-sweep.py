#!/usr/bin/env python3
"""生成 InstantID **受控调参扫描**工作流（weight × cn_strength 网格）。

为什么用生成器：扫描点数量由参数决定（3×3 就是 9 条子链），手写 YAML 无法循环；
生成器保证每个点的**其余变量完全一致**（同一底模/同一参考脸/同一 seed/同一提示词/
同一 latent），从而"差异只能来自这两个旋钮"——这是调参扫描可信的前提（dev-plan §P1）。

每个扫描点产出一条：Apply_wc → Sample_wc → Decode_wc → Save_wc(_wc) + FaceSim_wc
（`face-similarity` 对"生成图 vs 参考脸"打分 ⇒ 身份保真度自动量化，不用人眼比）。

用法：
  python3 nodes/_tools/gen-instantid-sweep.py --weights 0.5,0.8 --cn 0.3,0.8 \
      --out ~/.flowx-tmp/instantid-sweep.yaml [--steps 12] [--seed 20260926]
产出 YAML 可直接 upsert 到 Studio（~/.flowx-tmp/wfup.py）。
"""
import argparse
import pathlib

HEAD = '''Name: instantid-param-sweep
Description: >-
  InstantID 受控调参扫描（weight × cn_strength 网格）：同一 SDXL 底模 / 参考脸 / seed /
  提示词 / latent，只改身份强度 weight 与 ControlNet 强度 cn_strength，逐点落盘并用
  face-similarity 自动打分（身份保真度）。生成器：nodes/_tools/gen-instantid-sweep.py
Param:
  service_url: "https://44e44576.r10.vip.cpolar.cn"
  service_url_host: "https://44e44576.r10.vip.cpolar.cn"
  service_token: "30e494999da13134808ee280aea6dffd"
  ref_path: "~/flowx-output/test_person.jpg"
  seed: "{seed}"
  output_dir: "~/flowx-e2e-out"
Graph: |
  stateDiagram-v2
    [*] --> Ensure
    Ensure --> LoadCkpt
    Ensure --> LoadCN
    Ensure --> LoadIPA
    Ensure --> LoadRef
    Ensure --> EmptyLatent
    LoadRef --> FaceAnalyze
    LoadCkpt --> EncPos
    LoadCkpt --> EncNeg
{edges}
    Unload -->|done| [*]
Nodes:
  Ensure:
    name: Ensure
    config:
      nodeRef: inference-ensure@1.1.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        wait_seconds: 60
  LoadCkpt:
    name: LoadCkpt
    config:
      nodeRef: checkpoint-loader@1.3.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        service_url_host: "{{{{ Param.service_url_host }}}}"
        ckpt_name: "sd_xl_base_1.0"
        dtype: "fp16"
        offload: "model"
  LoadCN:
    name: LoadCN
    config:
      nodeRef: controlnet-load@1.1.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        service_url_host: "{{{{ Param.service_url_host }}}}"
        name: "instantid-controlnet"
  LoadIPA:
    name: LoadIPA
    config:
      nodeRef: ipadapter-load@1.1.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        service_url_host: "{{{{ Param.service_url_host }}}}"
        name: "instantid-ip-adapter"
  LoadRef:
    name: LoadRef
    config:
      nodeRef: load-image@1.4.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        service_url_host: "{{{{ Param.service_url_host }}}}"
        image_path: "{{{{ Param.ref_path }}}}"
  FaceAnalyze:
    name: FaceAnalyze
    config:
      nodeRef: instantid-face-analyze@1.2.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        service_url_host: "{{{{ Param.service_url_host }}}}"
        image: "{{{{ LoadRef.image }}}}"
        face_index: -1
        canvas_width: 512
        canvas_height: 512
        canvas_mode: "cover"
  EncPos:
    name: EncPos
    config:
      nodeRef: clip-text-encode@1.2.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        clip_ref: "{{{{ LoadCkpt.clip_ref }}}}"
        text: "portrait photo of a 30 year old woman, looking at camera, plain grey background, studio lighting, highly detailed, masterpiece, best quality"
  EncNeg:
    name: EncNeg
    config:
      nodeRef: clip-text-encode@1.2.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        clip_ref: "{{{{ LoadCkpt.clip_ref }}}}"
        text: "blurry, lowres, deformed, extra fingers, watermark, text"
  EmptyLatent:
    name: EmptyLatent
    config:
      nodeRef: empty-latent@1.1.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        width: 512
        height: 512
        batch_size: 1
'''

POINT = '''  Apply{wtag}:
    name: Apply w={w} cn={c}
    config:
      nodeRef: instantid-apply@1.1.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        service_url_host: "{{{{ Param.service_url_host }}}}"
        model: "{{{{ LoadCkpt.model_ref }}}}"
        ipadapter: "{{{{ LoadIPA.ipadapter }}}}"
        controlnet: "{{{{ LoadCN.control_net }}}}"
        face: "{{{{ FaceAnalyze.face }}}}"
        weight: {w}
        cn_strength: {c}
  Sample{wtag}:
    name: Sample w={w} cn={c}
    config:
      nodeRef: ksampler@1.5.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        service_url_host: "{{{{ Param.service_url_host }}}}"
        model_ref: "{{{{ Apply{wtag}.model }}}}"
        positive: "{{{{ EncPos.conditioning }}}}"
        negative: "{{{{ EncNeg.conditioning }}}}"
        latent: "{{{{ EmptyLatent.latent }}}}"
        seed: "{{{{ Param.seed }}}}"
        steps: {steps}
        cfg: 5.0
        sampler_name: "euler"
        scheduler: "normal"
        denoise: 1.0
        control: "{{{{ Apply{wtag}.control }}}}"
  Decode{wtag}:
    name: Decode w={w} cn={c}
    config:
      nodeRef: vae-decode@1.1.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        vae_ref: "{{{{ LoadCkpt.vae_ref }}}}"
        latent: "{{{{ Sample{wtag}.latent }}}}"
  Save{wtag}:
    name: Save w={w} cn={c}
    config:
      nodeRef: save-image@1.2.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        image: "{{{{ Decode{wtag}.image }}}}"
        filename_prefix: "iidp_w{w}_cn{c}"
        output_dir: "{{{{ Param.output_dir }}}}"
  Score{wtag}:
    name: Score w={w} cn={c}
    config:
      nodeRef: face-similarity@1.2.0
      executor: local
      params:
        service_url: "{{{{ Param.service_url }}}}"
        service_token: "{{{{ Param.service_token }}}}"
        service_url_host: "{{{{ Param.service_url_host }}}}"
        image_a: "{{{{ Decode{wtag}.image }}}}"
        image_b: "{{{{ LoadRef.image }}}}"
        soft: true
'''

UNLOAD = '''  Unload:
    name: Unload
    config:
      nodeRef: model-unload@1.1.0
      executor: local
      params:
        service_url: "{{ Param.service_url }}"
        service_token: "{{ Param.service_token }}"
'''


def tag(w, c):
    return ("w%s_c%s" % (w, c)).replace(".", "p")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="0.5,0.8")
    ap.add_argument("--cn", default="0.3,0.8")
    ap.add_argument("--steps", default="12")
    ap.add_argument("--seed", default="20260926")
    ap.add_argument("--out", default="instantid-sweep.yaml")
    a = ap.parse_args()
    weights = [w.strip() for w in a.weights.split(",") if w.strip()]
    cns = [c.strip() for c in a.cn.split(",") if c.strip()]

    edges, body = [], []
    for w in weights:
        for c in cns:
            t = tag(w, c)
            edges += [f"    FaceAnalyze --> Apply{t}",
                      f"    LoadCkpt --> Apply{t}",
                      f"    LoadIPA --> Apply{t}",
                      f"    LoadCN --> Apply{t}",
                      f"    Apply{t} --> Sample{t}",
                      f"    Sample{t} --> Decode{t}",
                      f"    Decode{t} --> Save{t}",
                      f"    Decode{t} --> Score{t}",
                      f"    FaceAnalyze --> Score{t}",
                      f"    Save{t} --> Unload",
                      f"    Score{t} --> Unload"]
            body.append(POINT.format(wtag=t, w=w, c=c, steps=a.steps))
    yaml_text = (HEAD.format(seed=a.seed, edges="\n".join(edges))
                 + "\n".join(body) + UNLOAD)
    out = pathlib.Path(a.out).expanduser()
    out.write_text(yaml_text)
    print(f"已生成 {out}：{len(weights)}×{len(cns)} = {len(weights)*len(cns)} 个扫描点"
          f"（weight={weights} × cn_strength={cns}, steps={a.steps}）")


if __name__ == "__main__":
    main()
