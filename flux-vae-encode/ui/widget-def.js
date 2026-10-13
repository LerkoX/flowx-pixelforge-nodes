const WIDGET_SPEC = {
  icon: '📥',
  title: 'FLUX VAE 编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'vae', label: 'vae（FLUX VAE 加载）', kind: 'text', mono: true },
    { key: 'image', label: 'image（待编码图像）', kind: 'text', mono: true },
    { key: 'poll_interval', label: '轮询间隔 poll_interval（秒）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '任务式调用（排队不占隧道、可取消）；图像 → latent；配采样器 denoise<1 做 img2img/hires-fix',
}
