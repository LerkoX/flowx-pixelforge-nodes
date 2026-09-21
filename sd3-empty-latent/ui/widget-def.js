const WIDGET_SPEC = {
  icon: '🧱',
  title: 'SD3 空 Latent',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'width', label: '宽（width）', kind: 'slider', min: 256, max: 1920, step: 64, default: 512 },
    { key: 'height', label: '高（16 的倍数）（height）', kind: 'slider', min: 256, max: 1920, step: 64, default: 512 },
    { key: 'batch_size', label: '批大小（batch_size）', kind: 'slider', min: 1, max: 8, step: 1, default: 1 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: 'SD3.5 空 Latent 画布',
}
