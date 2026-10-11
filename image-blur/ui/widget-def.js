const WIDGET_SPEC = {
  icon: '💨',
  title: '模糊',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'mode', label: '模糊类型', kind: 'select', options: ['gaussian', 'box', 'motion'], default: 'gaussian' },
    { key: 'radius', label: '半径 / 拖影长度', kind: 'slider', min: 0, max: 64, step: 1, default: 4 },
    { key: 'angle', label: '运动方向（度）', kind: 'slider', min: -180, max: 180, step: 1, default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
}
