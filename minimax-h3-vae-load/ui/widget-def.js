const WIDGET_SPEC = {
  icon: '🎛️',
  title: 'H3 VAE',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video_vae', label: '视频 VAE（diffusers 目录名）', kind: 'text', mono: true, default: 'vae-diffusers' },
    { key: 'audio_vae', label: '音频 VAE 官方文件名', kind: 'text', mono: true, default: 'minimax_h3_audio_vae_official.safetensors' },
    { key: 'dtype', label: '精度（bf16/fp16/fp32）', kind: 'text', default: 'bf16' },
    { key: 'device', label: '常驻设备（cuda/cpu）', kind: 'text', default: 'cuda' },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '双 VAE 组件（video + audio）→ h3-pipe',
}
