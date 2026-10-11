const WIDGET_SPEC = {
  icon: '🖊️',
  title: '线稿提取（三档）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像（上游 image 输出）', kind: 'text', mono: true },
    { key: 'variant', label: '线稿档位', kind: 'select', options: ['lineart', 'anime', 'standard'], default: 'lineart' },
    { key: 'detect_resolution', label: '检测分辨率', kind: 'slider', min: 256, max: 1024, step: 64, default: 512 },
    { key: 'coarse', label: '粗松线稿（仅 lineart 档）', kind: 'check', default: false },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['variant', 'detect_resolution', 'coarse'],
  compareInput: 'image',
  note: '黑底白线线稿 → controlnet-apply 的 hint（配 lineart / lineart_anime ControlNet）',
}
