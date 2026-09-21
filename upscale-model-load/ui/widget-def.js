const WIDGET_SPEC = {
  icon: '📦',
  title: '放大模型加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'name', label: '放大模型', kind: 'model', modelType: 'upscale' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '输出 upscale_model → upscale-model-apply；spandrel 自动识别架构与倍数',
}
