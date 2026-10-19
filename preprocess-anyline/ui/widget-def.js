const WIDGET_SPEC = {
  icon: '✏️',
  title: 'Anyline 高精度线稿（MTEED）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像（上游 image 输出）', kind: 'text', mono: true },
    { key: 'detect_resolution', label: '检测分辨率', kind: 'slider', min: 512, max: 1600, step: 64, default: 1280 },
    { key: 'guassian_sigma', label: '高斯去噪 sigma', kind: 'slider', min: 0, max: 6, step: 0.1, default: 2 },
    { key: 'intensity_threshold', label: '强度阈值', kind: 'slider', min: 1, max: 10, step: 1, default: 3 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
  note: '黑底白线高精度线稿 → controlnet-apply 的 hint（配 anyline/MistoLine ControlNet）',
}
