const WIDGET_SPEC = {
  icon: '⏱',
  title: '条件时间窗',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'cond', label: 'COND（接 CLIP 文本编码）', kind: 'text', mono: true },
    { key: 'start_percent', label: '起始进度（0~1）', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.0 },
    { key: 'end_percent', label: '结束进度（0~1）', kind: 'slider', min: 0, max: 1, step: 0.05, default: 1.0 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 300, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '条件只在采样进度 [start,end) 内生效（分段提示词）',
}
