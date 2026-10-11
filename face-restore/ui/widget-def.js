const WIDGET_SPEC = {
  icon: '🧑‍🦱',
  title: '人脸修复',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'model', label: '修复模型', kind: 'select', options: ['gfpgan', 'restoreformer'], default: 'gfpgan' },
    { key: 'upscale', label: '修复块放大', kind: 'slider', min: 1, max: 4, step: 0.5, default: 1.0 },
    { key: 'padding', label: '人脸框外扩比例', kind: 'slider', min: 0, max: 1.5, step: 0.05, default: 0.3 },
    { key: 'feather', label: '贴回羽化（px）', kind: 'slider', min: 0, max: 64, step: 2, default: 16 },
    { key: 'max_faces', label: '最多修几张脸', kind: 'slider', min: 1, max: 8, step: 1, default: 1 },
    { key: 'only_center', label: '只修画面中心最近的一张', kind: 'check', default: true },
    { key: 'det_thresh', label: '检测置信度阈值', kind: 'slider', min: 0.2, max: 0.9, step: 0.05, default: 0.5 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['model', 'upscale', 'padding', 'feather', 'max_faces', 'only_center', 'det_thresh'],
  compareInput: 'image',
  note: '与 detail-refine 的分工：只做修复增强（老照片/低清脸），不重新生成内容。GFPGAN 固定 512²（upscale 必须 1.0）；RestoreFormer 可 upscale=2',
};
