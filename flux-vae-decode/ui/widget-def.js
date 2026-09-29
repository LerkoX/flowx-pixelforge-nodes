const WIDGET_SPEC = {
  icon: '🖼',
  title: 'FLUX VAE 解码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'vae', label: 'vae（FLUX VAE 加载）', kind: 'text', mono: true },
    { key: 'latent', label: 'latent（FLUX 采样）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'packed latent → 图像；接保存图像或局部重绘',
}
