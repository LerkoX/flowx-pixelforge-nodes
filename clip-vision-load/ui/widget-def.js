const WIDGET_SPEC = {
  icon: '👁️',
  title: 'CLIP Vision 加载',
  fields: [
    {"key": "service_url", "label": "推理服务 service_url", "kind": "text", "mono": true, "placeholder": "http://…:8100"},
    {"key": "name", "label": "编码器模型", "kind": "model", "modelType": "clip_vision"},
    {"key": "service_token", "label": "service_token（可空）", "kind": "text", "mono": true},
  ],
  note: '输出 clip_vision → ipadapter-apply 节点',
}
