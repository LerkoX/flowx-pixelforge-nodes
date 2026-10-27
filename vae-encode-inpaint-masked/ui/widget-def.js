const WIDGET_SPEC = {
  icon: '🧊',
  title: 'VAE 编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'vae_ref', label: 'VAE 引用 ID（vae_ref）', kind: 'text', mono: true },
    { key: 'image', label: '待编码的图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'service_url_host', label: '画布可达地址 service_url_host（docker 用）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: '任务超时 job_timeout（秒）', kind: 'text', mono: true, advanced: true },
    { key: 'poll_interval', label: '轮询间隔 poll_interval（秒）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '将图像编码为 latent',
}
