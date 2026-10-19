const WIDGET_SPEC = {
  icon: '🌫️',
  title: '软边缘提取（PidiNet/HED）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像（上游 image 输出）', kind: 'text', mono: true },
    { key: 'variant', label: '检测器', kind: 'select', options: ['pidi', 'hed'], default: 'pidi' },
    { key: 'detect_resolution', label: '检测分辨率', kind: 'slider', min: 256, max: 1024, step: 64, default: 512 },
    { key: 'safe', label: 'safe 模式（更平滑）', kind: 'check', default: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
  note: '黑底白边软边缘 → controlnet-apply 的 hint（配 softedge ControlNet）',
}
