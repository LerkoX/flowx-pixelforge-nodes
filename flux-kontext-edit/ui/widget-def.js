const WIDGET_SPEC = {
  icon: '🎨',
  title: 'FLUX Kontext 图像编辑',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'model', label: 'MODEL', kind: 'text', mono: true },
    { key: 'vae', label: 'VAE', kind: 'text', mono: true },
    { key: 'cond', label: 'COND（编辑指令）', kind: 'text', mono: true },
    { key: 'image', label: '输入图片', kind: 'text', mono: true },
    { key: 'seed', label: '种子', kind: 'number', default: -1 },
    { key: 'steps', label: '步数', kind: 'slider', min: 4, max: 50, step: 1, default: 20 },
    { key: 'guidance', label: 'CFG', kind: 'slider', min: 1, max: 10, step: 0.5, default: 3.5 },
    { key: 'denoise', label: '去噪强度', kind: 'slider', min: 0.1, max: 1.0, step: 0.05, default: 0.7 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'FLUX.1-Kontext-dev 指令驱动图像编辑。用于参考图姿态重定向（与视频第一帧同姿态）。denoise 0.6~0.8 保留原图结构。',
}
