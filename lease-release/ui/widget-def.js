const WIDGET_SPEC = {
  icon: '🔓',
  title: '租约释放',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '显式释放本执行的推理服务租约',
}
