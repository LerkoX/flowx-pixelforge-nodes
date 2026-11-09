const WIDGET_SPEC = {
  icon: '🌊',
  title: 'Wan Transformer 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'transformer', label: '高噪 transformer 权重文件名', kind: 'text', mono: true, advanced: true },
    { key: 'transformer2', label: 'Wan2.2 低噪专家（留空=单专家）', kind: 'text', mono: true, advanced: true },
    { key: 'dtype', label: '精度（bf16/fp16/fp32）', kind: 'text', mono: true, advanced: false },
    { key: 'offload', label: 'offload（sequential/none）', kind: 'text', mono: true, advanced: false },
  ],
  note: 'Wan transformer 组件加载（14B bf16 逐层 offload）',
}
