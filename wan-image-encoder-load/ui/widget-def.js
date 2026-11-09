const WIDGET_SPEC = {
  icon: '🌊',
  title: 'Wan 图像编码器加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image_encoder', label: 'CLIP-ViT-H 权重文件名', kind: 'text', mono: true, advanced: true },
    { key: 'device', label: '设备（cuda/cpu）', kind: 'text', mono: true, advanced: false },
  ],
  note: 'Wan 图像编码器 CLIP-ViT-H（仅 i2v/animate 需要）',
}
