const WIDGET_SPEC = {
  icon: '🔗',
  title: 'Conditioning 拼接',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'cond_a', label: 'cond_a（第一段 COND）', kind: 'text', mono: true },
    { key: 'cond_b', label: 'cond_b（第二段 COND）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '两段提示词同时生效，可突破 77 token 截断；多区域构图串联 set_area 产物',
}
