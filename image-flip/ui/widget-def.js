const WIDGET_SPEC = {
  icon: '🪞',
  title: '翻转图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'image', label: '推理服务端的图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'mode', label: '翻转方向：horizontal（左右）/ ver（mode）', kind: 'select', options: ['horizontal', 'vertical'], default: 'horizontal' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '水平/垂直镜像翻转图像',
}
