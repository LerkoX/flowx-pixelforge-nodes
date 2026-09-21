const WIDGET_SPEC = {
  icon: '🎞️',
  title: 'VAE 视频解码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'vae', label: 'vae（VAE 对象）', kind: 'text', mono: true },
    { key: 'latents', label: 'latents（3D LATENT）', kind: 'text', mono: true },
    { key: 'num_frames', label: '帧数 num_frames（0=自动）', kind: 'text', default: 0 },
    { key: 'decode_chunk_size', label: '分块 decode_chunk_size', kind: 'slider', min: 1, max: 32, step: 1, default: 14 },
    { key: 'force_fp32', label: 'force_fp32（防过曝）', kind: 'check', default: true },
    { key: 'fps', label: '帧率 fps', kind: 'slider', min: 8, max: 60, step: 1, default: 24 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '3D latent → VIDEO；与 video-sample-latent 接力（采样/解码解耦）',
}
