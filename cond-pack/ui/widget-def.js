const WIDGET_SPEC = {
  icon: '🎛',
  title: '条件打包',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'cond', label: 'COND（接 CLIP 文本编码）', kind: 'text', mono: true },
    { key: 'weight', label: '整体权重', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'start_percent', label: '起始进度（0~1）', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.0 },
    { key: 'end_percent', label: '结束进度（0~1）', kind: 'slider', min: 0, max: 1, step: 0.05, default: 1.0 },
    { key: 'layer_start', label: '起始层（cross-attn）', kind: 'text', default: 0, advanced: true },
    { key: 'layer_end', label: '结束层（-1=末层）', kind: 'text', default: -1, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 300, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '把 COND 打包：权重 + 采样进度窗口 + cross-attn 层区间',
}
