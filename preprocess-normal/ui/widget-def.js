const WIDGET_SPEC = {
  icon: '🧊',
  title: '法线图（NormalBae）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像（上游 image 输出）', kind: 'text', mono: true },
    { key: 'detect_resolution', label: '检测分辨率', kind: 'slider', min: 256, max: 1024, step: 64, default: 512 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['detect_resolution'],
  compareInput: 'image',
  note: '法线图 → controlnet-apply 的 hint（配 normalbae ControlNet）',
}
