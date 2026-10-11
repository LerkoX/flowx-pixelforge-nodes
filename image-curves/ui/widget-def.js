const WIDGET_SPEC = {
  icon: '📈',
  title: '曲线映射',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'channel', label: '作用通道', kind: 'select', options: ['all', 'red', 'green', 'blue'], default: 'all' },
    { key: 'points', label: '曲线控制点', kind: 'curves', default: '0,0;255,255' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
}
