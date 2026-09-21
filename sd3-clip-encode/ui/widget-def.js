const WIDGET_SPEC = {
  icon: '📝',
  title: 'SD3 CLIP 编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'clip_ref', label: 'CLIP 引用 ID（SD3.5 管道视图）（clip_ref）', kind: 'text', mono: true },
    { key: 'text', label: '提示词文本（text）', kind: 'textarea' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: 'SD3.5 文本编码',
}
