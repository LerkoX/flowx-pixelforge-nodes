const WIDGET_SPEC = {
  icon: '🎞️',
  title: '视频末帧',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '视频对象 ID（video）', kind: 'text', mono: true },
    { key: 'offset', label: '末尾回退秒数（避开黑场尾帧）', kind: 'slider', min: 0, max: 2, step: 0.1, default: 0.2 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '抽最后一帧上传为 IMAGE 对象——末帧续接闭环（接 H3Gen.image）',
}
