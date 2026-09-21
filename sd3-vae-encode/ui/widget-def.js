const WIDGET_SPEC = {
  icon: '📥',
  title: 'SD3 VAE 编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'vae_ref', label: 'VAE 引用 ID（SD3.5 管道视图）（vae_ref）', kind: 'text', mono: true },
    { key: 'image', label: '输入图像引用（接 load-image 的输出）（image）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: 'SD3.5 VAE 编码：图像 → 16ch l',
}
