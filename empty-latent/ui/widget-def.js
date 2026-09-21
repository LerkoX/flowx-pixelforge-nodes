const WIDGET_SPEC = {
  icon: '⬜',
  title: '空 Latent 图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'width', label: '生成图像宽度（像素）（width）', kind: 'slider', min: 256, max: 1920, step: 64, default: 512 },
    { key: 'height', label: '生成图像高度（像素）（height）', kind: 'slider', min: 256, max: 1920, step: 64, default: 512 },
    { key: 'batch_size', label: '批大小（batch_size）', kind: 'slider', min: 1, max: 8, step: 1, default: 1 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '按宽高与 batch 在推理服务端创建空白 la',
}
