const WIDGET_SPEC = {
  icon: '🔄',
  title: '旋转图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'image', label: '推理服务端的图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'angle', label: '旋转角度（angle）', kind: 'slider', min: -180, max: 180, step: 1, default: 0 },
    { key: 'expand', label: 'true 时扩大画布容纳旋转后全图（expand）', kind: 'check', default: true },
    { key: 'resample', label: '重采样算法：nearest / bilinear（resample）', kind: 'select', options: ['bicubic', 'bilinear', 'nearest', 'lanczos'], default: 'bicubic' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '任意角度旋转图像',
}
