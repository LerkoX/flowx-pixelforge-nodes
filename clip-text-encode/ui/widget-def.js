const WIDGET_SPEC = {
  icon: '📝',
  title: 'CLIP 文本编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'clip_ref', label: 'CLIP 编码器引用 ID（clip_ref）', kind: 'text', mono: true },
    { key: 'text', label: '提示词文本（text）', kind: 'textarea' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '将提示词文本编码为 conditioning 张',
}
