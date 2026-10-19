const WIDGET_SPEC = {
  icon: '🔀',
  title: '内容打乱（Shuffle）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像（上游 image 输出）', kind: 'text', mono: true },
    { key: 'detect_resolution', label: '分块分辨率', kind: 'slider', min: 256, max: 1024, step: 64, default: 512 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
  note: '打乱块 → controlnet-apply 的 hint（配 shuffle ControlNet；保构图换内容）',
}
