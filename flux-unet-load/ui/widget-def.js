const WIDGET_SPEC = {
  icon: '🧱',
  title: 'FLUX UNET 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'transformer', label: 'transformer GGUF', kind: 'text', mono: true, default: 'flux1-schnell-Q4_K_S.gguf' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'GGUF Q4_K_S transformer（~6.8GB）；幂等，已常驻秒回',
}
