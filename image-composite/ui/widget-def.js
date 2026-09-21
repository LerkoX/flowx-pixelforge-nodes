const WIDGET_SPEC = {
  icon: '🧩',
  title: '图像合成',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'background', label: '底图图像对象 ID（background）', kind: 'text', mono: true },
    { key: 'foreground', label: '前景图像对象 ID（foreground）', kind: 'text', mono: true },
    { key: 'mask', label: '蒙版对象 ID（mask）', kind: 'text', mono: true },
    { key: 'x', label: '前景左上角 X 偏移（像素）（x）', kind: 'text', default: 0 },
    { key: 'y', label: '前景左上角 Y 偏移（像素）（y）', kind: 'text', default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '前景贴到底图',
}
