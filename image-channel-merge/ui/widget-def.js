const WIDGET_SPEC = {
  icon: '🧪',
  title: '通道合并',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'r_image', label: 'R 通道图像', kind: 'text', mono: true, default: '' },
    { key: 'g_image', label: 'G 通道图像', kind: 'text', mono: true, default: '' },
    { key: 'b_image', label: 'B 通道图像', kind: 'text', mono: true, default: '' },
    { key: 'alpha_image', label: 'alpha 通道图像（可空）', kind: 'text', mono: true, default: '' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '三路灰度图合并为 RGB/RGBA',
}
