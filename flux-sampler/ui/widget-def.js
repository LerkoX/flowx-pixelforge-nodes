const WIDGET_SPEC = {
  icon: '⚡',
  title: 'FLUX 采样器',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: 'service_url_host（预览用）', kind: 'text', mono: true, advanced: true },
    { key: 'model_ref', label: 'model_ref（FLUX 加载）', kind: 'text', mono: true },
    { key: 'cond', label: 'cond（FLUX 文本编码）', kind: 'text', mono: true },
    { key: 'width', label: '宽度', kind: 'select', options: ['512', '768', '1024'], default: '768' },
    { key: 'height', label: '高度', kind: 'select', options: ['512', '768', '1024'], default: '768' },
    { key: 'steps', label: '步数（schnell 1~4）', kind: 'slider', min: 1, max: 8, step: 1, default: 4 },
    { key: 'seed', label: 'seed（-1 随机）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'VAE 解码内置，image 直接接保存图像；GTX1080 建议 512~768 出稿',
}
