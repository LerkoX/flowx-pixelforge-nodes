const WIDGET_SPEC = {
  icon: '🎭',
  title: 'Latent 噪点蒙版',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'latent', label: 'latent（原图 vae-encode 结果）', kind: 'text', mono: true },
    { key: 'mask', label: 'mask（白=重绘区域）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: 'inpaint 核心：本节点 → sample（denoise<1）只重绘蒙版区域',
}
