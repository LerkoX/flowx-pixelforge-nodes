const WIDGET_SPEC = {
  icon: '🔧',
  title: 'H3 组装',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'transformer', label: 'TRANSFORMER 对象 ID', kind: 'text', mono: true },
    { key: 'text_encoder', label: 'TEXT_ENCODER 对象 ID', kind: 'text', mono: true },
    { key: 'vae', label: 'VAE 对象 ID', kind: 'text', mono: true },
    { key: 'shift_video', label: '视频 shift（官方 12.0）', kind: 'slider', min: 1, max: 24, step: 0.5, default: 12.0 },
    { key: 'shift_audio', label: '音频 shift（官方 3.0）', kind: 'slider', min: 1, max: 12, step: 0.5, default: 3.0 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 300 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '组件组装为 MODEL（register_components + 双 scheduler）',
}
