const WIDGET_SPEC = {
  icon: '🎯',
  title: 'SAM3 图像分割',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: '画布可达地址 service_url_host（docker 用）', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'prompt', label: '提示词（head / person / the head and hair）', kind: 'text', default: 'person' },
    { key: 'obj_pick', label: '目标选择', kind: 'select', options: ['all', 'best'], default: 'all' },
    { key: 'threshold', label: '置信阈值', kind: 'slider', min: 0.1, max: 0.9, step: 0.05, default: 0.5 },
    { key: 'dilate', label: '外扩 dilate（px）', kind: 'slider', min: 0, max: 40, step: 1, default: 0 },
    { key: 'pad', label: 'bbox 外扩 pad（px）', kind: 'slider', min: 0, max: 100, step: 5, default: 0 },
    { key: 'fill_holes', label: '孔洞填充 fill_holes（0/1）', kind: 'select', options: ['0', '1'], default: '0' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
  note: '抠头/头发：prompt=head + fill_holes=1（SAM3 会选中发丝但漏掉被包围的脸）；bbox 输出可接线给 image-crop 自动取景',
};
