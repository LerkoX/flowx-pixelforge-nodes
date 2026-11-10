const WIDGET_SPEC = {
  icon: '📝',
  title: 'Wan 文本编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'prompt', label: '正向提示词', kind: 'textarea', rows: 4 },
    { key: 'negative_prompt', label: '负向提示词（guidance>1 生效）', kind: 'textarea', rows: 3 },
    { key: 'max_sequence_length', label: 'max_sequence_length（0=pipe 值）', kind: 'text', default: 0 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'UMT5 编码 → COND（正+负两路）→ wan-gen',
}
