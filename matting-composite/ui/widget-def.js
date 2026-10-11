const WIDGET_SPEC = {
  icon: '🖼️',
  title: '抠图换底合成',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '前景图对象 ID（image，RGBA）', kind: 'text', mono: true },
    { key: 'mask', label: '前景蒙版（可空，留空用 alpha）', kind: 'text', mono: true },
    { key: 'bg_mode', label: '底色来源', kind: 'select', options: ['color', 'image'], default: 'color' },
    { key: 'bg_color', label: '纯色底色（#RRGGBB）', kind: 'text', mono: true, default: '#FFFFFF' },
    { key: 'bg_image', label: '背景图对象 ID（bg_mode=image）', kind: 'text', mono: true },
    { key: 'feather', label: '边缘羽化（px）', kind: 'slider', min: 0, max: 64, step: 1, default: 0 },
    { key: 'x', label: '前景偏移 X（px）', kind: 'slider', min: -512, max: 512, step: 8, default: 0 },
    { key: 'y', label: '前景偏移 Y（px）', kind: 'slider', min: -512, max: 512, step: 8, default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
  note: '白底商品主图：matting-birefnet → matting-refine → 本节点（bg_color=#FFFFFF）→ save-image',
};
