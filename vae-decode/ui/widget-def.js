const WIDGET_SPEC = {
  icon: '🌈',
  title: 'VAE 解码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'vae_ref', label: 'VAE 引用 ID（vae_ref）', kind: 'text', mono: true },
    { key: 'latent', label: '采样后的 latent 对象 ID（latent）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '将采样后的 latent 解码为图像',
}
