const WIDGET_SPEC = {
  icon: '🌈',
  title: 'FLUX VAE 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'name', label: 'VAE 目录名', kind: 'text', mono: true, default: 'vae' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'FLUX 专用 AutoencoderKL（fp16，~0.3GB）',
}
