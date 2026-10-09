const WIDGET_SPEC = {
  icon: '🧬',
  title: 'FLUX Redux 参考图',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'cond', label: 'cond（FLUX 文本编码）', kind: 'text', mono: true },
    { key: 'image', label: 'image（load-image 参考图）', kind: 'text', mono: true },
    { key: 'strength', label: '参考图强度（1=标准）', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'reducer', label: 'reducer 权重文件名', kind: 'text', mono: true, default: 'flux1-redux-dev.safetensors', advanced: true },
    { key: 'vision', label: 'vision 编码器目录', kind: 'text', mono: true, default: 'siglip-so400m-patch14-384', advanced: true },
    { key: 'release', label: 'release（用后释放）', kind: 'text', default: 0, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 900, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '输出 cond 接 flux-sampler；参考图构图/主体/风格注入文本条件序列',
}
