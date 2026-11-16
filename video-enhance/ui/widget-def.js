const WIDGET_SPEC = {
  icon: '✨',
  title: 'Video Enhance（二次修复）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '输入视频（接 wan-animate2 / load-video）', kind: 'text', mono: true },
    { key: 'face_model', label: '面部修复模型', kind: 'select', options: ['gfpgan', 'restoreformer', 'none'], default: 'gfpgan' },
    { key: 'upscale_model', label: '超分模型（/models/upscale 下文件名，空=关）', kind: 'text', default: '' },
    { key: 'det_thresh', label: '人脸检测阈值', kind: 'slider', min: 0.1, max: 0.9, step: 0.05, default: 0.5, advanced: true },
    { key: 'face_padding', label: '人脸框外扩', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.3, advanced: true },
    { key: 'face_feather', label: '贴回羽化 px', kind: 'slider', min: 0, max: 48, step: 2, default: 16, advanced: true },
    { key: 'face_weight', label: '修复混合权重（低保身份高修得狠）', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.5 },
    { key: 'tile', label: '超分分块（0=整图）', kind: 'text', default: 0, advanced: true },
    { key: 'fps', label: '输出帧率（0=跟随输入）', kind: 'text', default: 0, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '逐帧 GFPGAN 面部修复 + RealESRGAN 超分；高速运动场景可能轻微闪烁',
}
