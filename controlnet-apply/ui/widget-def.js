const WIDGET_SPEC = {
  icon: '🎛️',
  title: 'ControlNet 应用',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'control_net', label: 'control_net（controlnet-load 输出）', kind: 'text', mono: true },
    { key: 'image', label: 'hint 图（预处理器输出线稿）', kind: 'text', mono: true },
    { key: 'strength', label: '强度 strength', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'start_percent', label: '生效起点 start_percent', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.0 },
    { key: 'end_percent', label: '生效终点 end_percent', kind: 'slider', min: 0, max: 1, step: 0.05, default: 1.0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  replay: ['strength', 'start_percent', 'end_percent'],
  note: '输出 control → 采样节点 control 端口；strength=0 ≡ 关闭',
}
