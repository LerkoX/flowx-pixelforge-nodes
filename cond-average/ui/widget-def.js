const WIDGET_SPEC = {
  icon: '⚖️',
  title: 'Conditioning 平均',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'cond_a', label: 'cond_a（权重 weight 侧）', kind: 'text', mono: true },
    { key: 'cond_b', label: 'cond_b（权重 1-weight 侧）', kind: 'text', mono: true },
    { key: 'weight', label: 'cond_a 权重 weight', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.5 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: 'out = a*weight + b*(1-weight)；风格/强度插值用',
}
