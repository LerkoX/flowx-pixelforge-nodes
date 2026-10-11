const WIDGET_SPEC = {
  icon: '🎨',
  title: '色相饱和度明度',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'hue', label: '色相偏移（度）', kind: 'slider', min: -180, max: 180, step: 1, default: 0 },
    { key: 'saturation', label: '饱和度', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'lightness', label: '明度偏移', kind: 'slider', min: -1, max: 1, step: 0.05, default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
}
