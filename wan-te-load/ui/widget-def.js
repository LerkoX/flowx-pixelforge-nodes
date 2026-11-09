const WIDGET_SPEC = {
  icon: '🌊',
  title: 'Wan 文本塔加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'text_encoder', label: 'UMT5-XXL 权重文件名', kind: 'text', mono: true, advanced: true },
    { key: 'dtype', label: '精度（bf16/fp16/fp32）', kind: 'text', mono: true, advanced: false },
    { key: 'offload', label: 'offload（sequential/none）', kind: 'text', mono: true, advanced: false },
  ],
  note: 'Wan 文本编码器 UMT5-XXL（11GB bf16）',
}
