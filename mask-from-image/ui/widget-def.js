const WIDGET_SPEC = {
  icon: '🎭',
  title: '图像转蒙版',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'image', label: '推理服务端的图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'channel', label: '提取通道：luminance（亮度）/ red （channel）', kind: 'select', options: ['luminance', 'red', 'green', 'blue'], default: 'luminance' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '从图像提取蒙版',
}
