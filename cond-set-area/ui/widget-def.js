const WIDGET_SPEC = {
  icon: '📐',
  title: 'Conditioning 设区域',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'cond', label: 'cond（COND 对象）', kind: 'text', mono: true },
    { key: 'x', label: 'x（图像像素）', kind: 'text', default: 0 },
    { key: 'y', label: 'y（图像像素）', kind: 'text', default: 0 },
    { key: 'width', label: 'width（图像像素）', kind: 'text', default: 512 },
    { key: 'height', label: 'height（图像像素）', kind: 'text', default: 512 },
    { key: 'strength', label: '区域权重 strength', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '多区域构图：多个 set_area 经 cond-combine 拼接；重叠区域效果叠加',
}
