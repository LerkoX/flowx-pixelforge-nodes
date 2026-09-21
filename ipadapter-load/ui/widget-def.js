const WIDGET_SPEC = {
  icon: '🧩',
  title: 'IPAdapter 加载',
  fields: [
    {"key": "service_url", "label": "推理服务 service_url", "kind": "text", "mono": true, "placeholder": "http://…:8100"},
    {"key": "name", "label": "适配器权重", "kind": "model", "modelType": "ipadapter"},
    {"key": "service_token", "label": "service_token（可空）", "kind": "text", "mono": true},
  ],
  note: '输出 ipadapter → ipadapter-apply 节点',
}
