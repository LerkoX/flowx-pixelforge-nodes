const WIDGET_SPEC = {
  icon: '📝',
  title: 'CLIP 文本编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'clip_ref', label: 'CLIP 编码器引用 ID（clip_ref）', kind: 'text', mono: true },
    { key: 'text', label: '提示词文本（text）', kind: 'textarea' },
    { key: 'width', label: 'SDXL 原始宽度（width）', kind: 'slider', min: 0, max: 2048, step: 64, default: 0 },
    { key: 'height', label: 'SDXL 原始高度（height）', kind: 'slider', min: 0, max: 2048, step: 64, default: 0 },
    { key: 'clip_skip', label: '跳过末 n 层文本编码（clip_skip）', kind: 'slider', min: -4, max: 0, step: 1, default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '将提示词文本编码为 conditioning',
}
