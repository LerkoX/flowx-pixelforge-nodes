const WIDGET_SPEC = {
  icon: '📝',
  title: 'FLUX 文本编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'clip', label: 'clip（FLUX DualCLIP 加载）', kind: 'text', mono: true },
    { key: 'text', label: '提示词（自然语言长句）', kind: 'textarea', placeholder: 'a cozy coffee shop ...' },
    { key: 'max_seq', label: 'T5 最大序列长度', kind: 'select', options: ['256', '512'], default: '256', advanced: true },
    { key: 'release_t5', label: '编码后卸载 T5', kind: 'select', options: ['1', '0'], default: '1', advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'T5-XXL+CLIP-L → COND；FLUX 免负向提示词；首次建缓存约 6 分钟',
}
