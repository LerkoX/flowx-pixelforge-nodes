const WIDGET_SPEC = {
  icon: '🖼️',
  title: '扩画布（Outpainting）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'left', label: '左扩边（像素）', kind: 'slider', min: 0, max: 1024, step: 8, default: 0 },
    { key: 'right', label: '右扩边（像素）', kind: 'slider', min: 0, max: 1024, step: 8, default: 256 },
    { key: 'top', label: '上扩边（像素）', kind: 'slider', min: 0, max: 1024, step: 8, default: 0 },
    { key: 'bottom', label: '下扩边（像素）', kind: 'slider', min: 0, max: 1024, step: 8, default: 0 },
    { key: 'fill', label: '新扩区域底色', kind: 'select', options: ['edge', 'black', 'white', 'gray'], default: 'edge' },
    { key: 'feather', label: '蒙版边界羽化（像素）', kind: 'slider', min: 0, max: 256, step: 4, default: 32 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
  note: '输出 image（扩边图）+ mask（扩图蒙版）；接 latent-set-noise-mask + ksampler(denoise=1.0)',
}
