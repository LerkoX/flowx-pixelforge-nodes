const WIDGET_SPEC = {
  icon: '🎭',
  title: '人物遮罩（SAM3）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '源视频对象 ID', kind: 'text', mono: true },
    { key: 'prompt', label: '目标人物描述（英文更稳）', kind: 'text', default: 'person' },
    { key: 'obj_pick', label: '多目标策略', kind: 'select', options: ['all', 'best'], default: 'all', advanced: true },
    { key: 'dilate', label: '遮罩膨胀像素', kind: 'slider', min: 0, max: 40, step: 1, default: 12 },
    { key: 'max_frames', label: '最多帧数（0=全片）', kind: 'number', default: 0, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'SAM3 检测+全片追踪 → 灰度遮罩视频（255=选中人物），接 H3Gen.inpaint_mask',
}
