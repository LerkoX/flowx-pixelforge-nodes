const WIDGET_SPEC = {
  icon: '🎚️',
  title: '亮度对比度伽马锐度',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'brightness', label: '亮度', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'contrast', label: '对比度', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'gamma', label: '伽马', kind: 'slider', min: 0.1, max: 3, step: 0.05, default: 1.0 },
    { key: 'sharpness', label: '锐度', kind: 'slider', min: 0, max: 3, step: 0.05, default: 1.0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
}
