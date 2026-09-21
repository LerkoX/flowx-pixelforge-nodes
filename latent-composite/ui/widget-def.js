const WIDGET_SPEC = {
  icon: '🧩',
  title: 'Latent 合成',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'dst', label: 'dst（底图 LATENT）', kind: 'text', mono: true },
    { key: 'src', label: 'src（贴入 LATENT）', kind: 'text', mono: true },
    { key: 'x', label: 'x（图像像素）', kind: 'text', default: 0 },
    { key: 'y', label: 'y（图像像素）', kind: 'text', default: 0 },
    { key: 'feather', label: '羽化 feather（图像像素）', kind: 'slider', min: 0, max: 128, step: 1, default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '越界自动裁剪；先 composite 再 set_noise_mask（反之会被拒绝）',
}
