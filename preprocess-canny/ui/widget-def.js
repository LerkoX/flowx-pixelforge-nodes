const WIDGET_SPEC = {
  icon: '✏️',
  title: 'Canny 线稿提取',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'image', label: '输入图像（上游 image 输出）', kind: 'text', mono: true },
    { key: 'low_threshold', label: '低阈值 low_threshold', kind: 'slider', min: 0, max: 255, step: 1, default: 100 },
    { key: 'high_threshold', label: '高阈值 high_threshold', kind: 'slider', min: 0, max: 255, step: 1, default: 200 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  replay: ['low_threshold', 'high_threshold'],
  compareInput: 'image',
  note: '黑底白线线稿 → ControlNet canny 的 hint；调低阈值线稿更密',
}
