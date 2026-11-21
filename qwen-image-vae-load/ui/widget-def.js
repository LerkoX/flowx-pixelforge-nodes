const WIDGET_SPEC = {
  icon: '🎛️',
  title: 'Qwen-Image VAE 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'model', label: '官方仓库目录名', kind: 'text', mono: true, advanced: true },
    { key: 'vae', label: 'VAE 单文件（留空=目录）', kind: 'text', mono: true, advanced: true },
    { key: 'device', label: '设备（cuda/cpu）', kind: 'text', mono: true, advanced: false },
  ],
  note: 'Qwen-Image VAE 加载（fp32）',
}
