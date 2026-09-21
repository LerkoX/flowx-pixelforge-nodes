const WIDGET_SPEC = {
  icon: '🕸️',
  title: 'ControlNet 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, placeholder: 'http://…:8100' },
    { key: 'name', label: 'ControlNet 模型', kind: 'model', modelType: 'controlnet' },
    { key: 'dtype', label: '精度 dtype', kind: 'select', options: ['auto', 'fp16', 'bf16', 'fp32'], default: 'auto' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '输出 control_net → controlnet-apply 节点',
}
