const WIDGET_SPEC = {
  icon: '🔎',
  title: '图像放大',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'image', label: '待放大的图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'scale', label: '放大倍率（默认 2.0）（scale）', kind: 'slider', min: 1, max: 4, step: 0.5, default: 2.0 },
    { key: 'width', label: '目标宽度（像素）（width）', kind: 'text', default: 0 },
    { key: 'height', label: '目标高度（像素）（height）', kind: 'text', default: 0 },
    { key: 'method', label: '重采样算法：lanczos（method）', kind: 'select', options: ['lanczos', 'bicubic', 'bilinear', 'nearest'], default: 'lanczos' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '图像放大',
}
