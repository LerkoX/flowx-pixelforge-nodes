const WIDGET_SPEC = {
  icon: '🌈',
  title: 'VAE 加载（外挂）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'name', label: 'VAE 模型', kind: 'model', modelType: 'vae' },
    { key: 'dtype', label: '精度 dtype', kind: 'select', options: ['auto', 'fp16', 'fp32'], default: 'auto' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '输出 vae → vae-decode/vae-encode；auto=fp32（外挂 VAE 的意义即解码质量）',
}
