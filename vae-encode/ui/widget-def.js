const WIDGET_SPEC = {
  icon: '🧊',
  title: 'VAE 编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'vae_ref', label: 'VAE 引用 ID（vae_ref）', kind: 'text', mono: true },
    { key: 'image', label: '待编码的图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '将图像编码为 latent',
}
