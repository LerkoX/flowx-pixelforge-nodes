const WIDGET_SPEC = {
  icon: '🔪',
  title: '锐化',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'mode', label: '锐化类型', kind: 'select', options: ['usm', 'simple'], default: 'usm' },
    { key: 'radius', label: 'USM 半径', kind: 'slider', min: 0, max: 32, step: 1, default: 2 },
    { key: 'percent', label: '强度（%）', kind: 'slider', min: 0, max: 300, step: 5, default: 150 },
    { key: 'threshold', label: 'USM 阈值', kind: 'slider', min: 0, max: 255, step: 1, default: 3 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
}
