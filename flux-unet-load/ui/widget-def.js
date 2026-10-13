const WIDGET_SPEC = {
  icon: '🧱',
  title: 'FLUX UNET 加载',
  fields: [
    { key: 'service_url_host', label: '画布可达地址 service_url_host（docker 用）', kind: 'text', mono: true, advanced: true },
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'transformer', label: 'transformer GGUF', kind: 'text', mono: true, default: 'flux1-schnell-Q4_K_S.gguf' },
    { key: 'dtype', label: 'compute 精度（dtype）', kind: 'select', options: ['auto', 'fp16', 'bf16', 'fp32'], default: 'auto' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒，异步任务超时）', kind: 'text', default: 1800, advanced: true },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5, advanced: true },
  ],
  note: 'GGUF Q4_K_S transformer（~6.8GB）；幂等，已常驻秒回；异步 /jobs 轮询，隧道下也稳',
}
