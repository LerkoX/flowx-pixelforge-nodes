const WIDGET_SPEC = {
  icon: '🩺',
  title: '推理服务健康检查',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'wait_seconds', label: '服务未就绪时的最长等待秒数（wait_seconds）', kind: 'text', default: 60 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '检查文生图推理服务',
}
