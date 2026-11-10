const WIDGET_SPEC = {
  icon: '➕',
  title: '条件并列',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'cond_a', label: 'COND A', kind: 'text', mono: true },
    { key: 'cond_b', label: 'COND B（可空）', kind: 'text', mono: true, default: '' },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 300, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '多条条件并列（各自保留区域/权重/时间窗）',
}
