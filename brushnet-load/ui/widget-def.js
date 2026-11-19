const WIDGET_SPEC = {
  icon: '🖌️',
  title: 'BrushNet 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, placeholder: 'http://…:8100' , advanced: true},
    { key: 'service_url_host', label: '画布侧服务地址（service_url_host）', kind: 'text', mono: true , advanced: true},
    { key: 'name', label: 'BrushNet 权重名（MODELS_DIR/brushnet/ 下）', kind: 'text', mono: true, placeholder: 'brushnet_random_mask' },
    { key: 'dtype', label: '精度 dtype', kind: 'select', options: ['auto', 'fp16', 'bf16', 'fp32'], default: 'auto' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '输出 control_net → brushnet-apply 节点；配套权重 Sanster/brushnet_random_mask 或 PowerPaint v2 的 PowerPaint_Brushnet',
}
