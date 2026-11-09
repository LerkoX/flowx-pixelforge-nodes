const WIDGET_SPEC = {
  icon: '🌊',
  title: 'Wan VAE 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'vae', label: 'Wan VAE 权重文件名', kind: 'text', mono: true, advanced: true },
    { key: 'device', label: '设备（cuda/cpu）', kind: 'text', mono: true, advanced: false },
  ],
  note: 'Wan VAE（强制 fp32，解码质量更好）',
}
