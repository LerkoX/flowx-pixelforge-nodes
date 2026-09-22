const WIDGET_SPEC = {
  icon: '🧹',
  title: '显存卸载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'target', label: '目标模型（留空=全部卸载）', kind: 'text', mono: true, placeholder: 'majicmixRealistic_v7 / 空' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '视频/大模型段跑完先卸载再跑图像，避免显存挤压导致采样退化到分钟级/步；在用（正被别的任务使用）的模型不会被卸',
}
