const WIDGET_SPEC = {
  icon: '🧬',
  title: '通道分离',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'channel', label: '导出通道', kind: 'select', options: ['luminance', 'red', 'green', 'blue', 'alpha'], default: 'luminance' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
}
