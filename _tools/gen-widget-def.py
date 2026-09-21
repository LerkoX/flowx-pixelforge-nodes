#!/usr/bin/env python3
"""gen-widget-def.py：老节点 → widget-base 公共件批量迁移生成器。

读各节点 flowx.json 的 parameters，按类型映射 + 手工调优表生成
ui/widget-def.js，随后用 build-widget.py 拼接为单文件 node-widget.js。

用法：python3 _tools/gen-widget-def.py            # 全量（29 个老节点）
      python3 _tools/gen-widget-def.py ksampler   # 指定节点
类型映射默认规则：
  boolean → check；integer/float → text（在 SLIDER 表中 → slider）；
  string  → text（在 SELECT/TEXTAREA/MODEL 表中优先）；引用/路径类 → mono
手工调优表（select 取值均已核对节点/服务端实现）：
  MODEL   checkpoint-loader.ckpt_name / lora-loader.lora_name / motion-loader.motion_name
  SLIDER  采样步数/cfg/denoise、缩放、蒙版半径、latent 尺寸等
  SELECT  image-flip.mode / mask-from-image.channel / image-rotate.resample / image-upscale.method
  TEXTAREA clip 提示词
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

NODES = [
    'checkpoint-loader', 'clip-text-encode', 'detail-refine', 'empty-latent',
    'image-composite', 'image-crop', 'image-flip', 'image-rotate',
    'image-upscale', 'inference-ensure', 'ksampler', 'load-image',
    'lora-loader', 'mask-feather', 'mask-from-image', 'mask-grow',
    'mask-invert', 'mask-to-image', 'motion-loader', 'save-image',
    'save-video', 'sd3-clip-encode', 'sd3-empty-latent', 'sd3-sampler',
    'sd3-vae-decode', 'sd3-vae-encode', 'vae-decode', 'vae-encode',
    'video-gen',
]

MODEL = {
    'checkpoint-loader': {'ckpt_name': 'checkpoint'},
    'lora-loader': {'lora_name': 'lora'},
    'motion-loader': {'motion_name': 'motion'},
}
SLIDER = {
    'ksampler': {'steps': (1, 150, 1), 'cfg': (1, 30, 0.5), 'denoise': (0, 1, 0.01)},
    'sd3-sampler': {'steps': (1, 150, 1), 'cfg': (1, 30, 0.5), 'denoise': (0, 1, 0.01)},
    'video-gen': {'steps': (1, 150, 1), 'cfg': (1, 30, 0.5)},
    'detail-refine': {'steps': (1, 150, 1), 'cfg': (1, 30, 0.5),
                      'denoise': (0, 1, 0.01), 'conf': (0, 1, 0.05), 'padding': (0, 1, 0.05)},
    'image-upscale': {'scale': (1, 4, 0.5)},
    'mask-feather': {'radius': (0, 64, 1)},
    'mask-grow': {'radius': (-64, 64, 1)},
    'empty-latent': {'width': (256, 1920, 64), 'height': (256, 1920, 64), 'batch_size': (1, 8, 1)},
    'sd3-empty-latent': {'width': (256, 1920, 64), 'height': (256, 1920, 64), 'batch_size': (1, 8, 1)},
    'image-rotate': {'angle': (-180, 180, 1)},
}
SAMPLER_NAMES = ['euler', 'euler_a', 'ddim', 'lms', 'dpmpp_2m', 'dpmpp_2m_sde', 'uni_pc']
SCHEDULERS = ['normal', 'karras', 'exponential', 'beta']
SELECT = {
    'ksampler': {'sampler_name': SAMPLER_NAMES, 'scheduler': SCHEDULERS},
    'sd3-sampler': {'sampler_name': SAMPLER_NAMES, 'scheduler': SCHEDULERS},
    'detail-refine': {'sampler_name': SAMPLER_NAMES, 'scheduler': SCHEDULERS},
    'image-flip': {'mode': ['horizontal', 'vertical']},
    'mask-from-image': {'channel': ['luminance', 'red', 'green', 'blue']},
    'image-rotate': {'resample': ['bicubic', 'bilinear', 'nearest', 'lanczos']},
    'image-upscale': {'method': ['lanczos', 'bicubic', 'bilinear', 'nearest']},
}
TEXTAREA = {
    'clip-text-encode': ['text'],
    'sd3-clip-encode': ['text'],
}
LABEL = {
    'service_url': '推理服务 service_url',
    'service_token': 'service_token（可空）',
    'preview_every': 'preview_every（0=关）',
    'job_timeout': 'job_timeout（秒）',
    'poll_interval': 'poll_interval（秒）',
}
MONO_NAMES = {
    'service_url', 'service_token', 'model_ref', 'image', 'mask', 'latent',
    'positive', 'negative', 'prompt', 'negative_prompt', 'background',
    'foreground', 'video', 'image_path', 'output_dir', 'filename_prefix',
}


def short_desc(desc: str) -> str:
    for sep in ('，', '。', ',', '；', ';'):
        if sep in desc:
            desc = desc.split(sep)[0]
    desc = desc[:24]
    # 截断可能留下未闭合括号，剥掉尾部悬空半括号
    while desc.count('（') > desc.count('）') and desc.endswith('(' + '' or '（'):
        desc = desc[:-1]
    if desc.count('（') > desc.count('）'):
        desc = desc[:desc.rfind('（')]
    return desc


def field_of(node: str, p: dict) -> dict:
    name, ptype = p['name'], p.get('type', 'string')
    desc = short_desc(p.get('description', ''))
    label = LABEL.get(name) or (f'{desc}（{name}）' if desc else name)
    f = {'key': name, 'label': label}
    if name in MODEL.get(node, {}):
        f.update(kind='model', modelType=MODEL[node][name])
        return f
    if name in SELECT.get(node, {}):
        f.update(kind='select', options=SELECT[node][name],
                 default=str(p.get('default', SELECT[node][name][0])))
        return f
    if name in SLIDER.get(node, {}):
        lo, hi, step = SLIDER[node][name]
        f.update(kind='slider', min=lo, max=hi, step=step)
        if p.get('default') not in (None, ''):
            f['default'] = p['default']
        return f
    if name in TEXTAREA.get(node, []):
        f['kind'] = 'textarea'
        return f
    if ptype == 'boolean':
        f.update(kind='check', default=bool(p.get('default', False)))
        return f
    f['kind'] = 'text'
    if name in MONO_NAMES or name.endswith('_ref'):
        f['mono'] = True
    if name in ('service_url', 'service_token'):
        f['advanced'] = True  # 连接参数收进折叠区，保持节点紧凑
    if p.get('default') not in (None, ''):
        f['default'] = p['default']
    return f


def js_val(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return '[' + ', '.join(js_val(x) for x in v) + ']'
    return "'" + str(v).replace("'", "\\'") + "'"


def gen(node: str) -> bool:
    ndir = ROOT / node
    meta = json.load(open(ndir / 'flowx.json'))
    params = meta.get('parameters', [])
    fields = [field_of(node, p) for p in params]
    lines = ['const WIDGET_SPEC = {']
    lines.append(f"  icon: {js_val(meta.get('icon') or '🧩')},")
    lines.append(f"  title: {js_val(meta.get('displayName') or node)},")
    lines.append('  fields: [')
    for f in fields:
        body = ', '.join(f'{k}: {js_val(v)}' for k, v in f.items())
        lines.append('    { ' + body + ' },')
    lines.append('  ],')
    if any(p['name'] == 'preview_every' for p in params):
        lines.append("  previewEveryKey: 'preview_every',")
    desc = short_desc(meta.get('description', ''))
    if desc:
        lines.append(f'  note: {js_val(desc)},')
    lines.append('}')
    ui = ndir / 'ui'
    ui.mkdir(exist_ok=True)
    (ui / 'widget-def.js').write_text('\n'.join(lines) + '\n')
    return True


def main():
    targets = sys.argv[1:] or NODES
    for n in targets:
        if not (ROOT / n / 'flowx.json').is_file():
            print(f'{n}: 无 flowx.json，跳过', file=sys.stderr)
            continue
        gen(n)
        print(f'{n}: -> ui/widget-def.js')


if __name__ == '__main__':
    main()
