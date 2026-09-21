const WIDGET_SPEC = {
  icon: '🔍',
  title: 'Latent 放大',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'latent', label: 'latent（LATENT 对象）', kind: 'text', mono: true },
    { key: 'scale', label: '倍率 scale', kind: 'slider', min: 1, max: 4, step: 0.5, default: 2.0 },
    { key: 'width', label: '目标宽 width（0=按倍率）', kind: 'text', default: 0 },
    { key: 'height', label: '目标高 height（0=按倍率）', kind: 'text', default: 0 },
    { key: 'method', label: '插值 method', kind: 'select', options: ['bicubic', 'bilinear', 'nearest'], default: 'bicubic' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: 'hires.fix 潜空间路径：放大后接 sample（denoise 0.3~0.5）精修',
}
