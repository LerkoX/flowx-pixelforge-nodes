const WIDGET_SPEC = {
  icon: '🌫️',
  title: '蒙版羽化',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'mask', label: '推理服务端的蒙版对象 ID（灰度图像）（mask）', kind: 'text', mono: true },
    { key: 'radius', label: '高斯模糊半径（radius）', kind: 'slider', min: 0, max: 64, step: 1, default: 8 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '蒙版高斯羽化',
}
