const WIDGET_SPEC = {
  icon: '🩹',
  title: '脸部修复回贴（FaceFix）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '源视频对象 ID（与 face-crop 同源）', kind: 'text', mono: true },
    { key: 'crop_video', label: '重绘后特写视频（H3Gen.video）', kind: 'text', mono: true },
    { key: 'boxes', label: 'boxes JSON（face-crop.boxes）', kind: 'text', mono: true },
    { key: 'feather', label: '羽化像素（贴回边缘过渡）', kind: 'slider', min: 0, max: 96, step: 4, default: 24 },
    { key: 'tone_mode', label: '色调口径', kind: 'select', options: ['freq_sep', 'ema', 'none'], default: 'freq_sep' },
    { key: 'lowpass_frac', label: 'freq_sep 低通（脸高比例）', kind: 'slider', min: 0.04, max: 0.25, step: 0.01, default: 0.1 },
    { key: 'detail_gain', label: '细节增益', kind: 'slider', min: 0.5, max: 1.5, step: 0.05, default: 1 },
    { key: 'full_fix_below', label: '全修复脸高上限 px（0=关渐隐）', kind: 'number', default: 0, advanced: true },
    { key: 'no_fix_above', label: '不修复脸高下限 px', kind: 'number', default: 0, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'number', default: 1800, advanced: true },
  ],
  note: 'PixelAI H3 FaceFix ③：重绘裁片按 boxes 缩放贴回原视频。color_match 把 patch 外环色温/曝光对齐源帧（治蓝色方块闪动）；Face Size Fade：远脸全修近脸不动；音轨自动透传',
}
