const WIDGET_SPEC = {
  icon: '⚡',
  title: 'Wan 文本向量缓存',
  fields: [
    { key: 'prompt', label: '正向提示词', kind: 'textarea', rows: 4 },
    { key: 'negative_prompt', label: '负向提示词（guidance>1 生效）', kind: 'textarea', rows: 3 },
    { key: 'max_sequence_length', label: 'max_sequence_length', kind: 'text', default: 512 },
    { key: 'text_encoder', label: '文本编码器目录（空=默认）', kind: 'text', mono: true, advanced: true },
    { key: 'service_url', label: 'service_url', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '首算一次落盘，之后跳过 11GB UMT5 → 接 wan-animate2 的 prompt_embeds_path',
}
