const WIDGET_SPEC = {
  icon: '📐',
  title: 'M-LSD 直线检测',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像（上游 image 输出）', kind: 'text', mono: true },
    { key: 'detect_resolution', label: '检测分辨率', kind: 'slider', min: 256, max: 1024, step: 64, default: 512 },
    { key: 'thr_v', label: '竖线阈值', kind: 'slider', min: 0.05, max: 0.5, step: 0.05, default: 0.1 },
    { key: 'thr_d', label: '横线阈值', kind: 'slider', min: 0.05, max: 0.5, step: 0.05, default: 0.1 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['detect_resolution', 'thr_v', 'thr_d'],
  compareInput: 'image',
  note: '黑底白直线 → controlnet-apply 的 hint（配 mlsd ControlNet）',
}
