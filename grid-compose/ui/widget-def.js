const WIDGET_SPEC = {
  icon: '🔳',
  title: '朋友圈九宫格',
  fields: [
    { key: 'images', label: '9 张图像对象 ID（行优先）', kind: 'text', mono: true },
    { key: 'cell_size', label: '每格边长（px）', kind: 'slider', min: 256, max: 1024, step: 32, default: 640 },
    { key: 'gap', label: '格子间隙（px）', kind: 'slider', min: 0, max: 64, step: 1, default: 10 },
    { key: 'margin', label: '外圈边距（px）', kind: 'slider', min: 0, max: 128, step: 1, default: 10 },
    { key: 'radius', label: '格子圆角（px）', kind: 'slider', min: 0, max: 80, step: 1, default: 0 },
    { key: 'bg', label: '背景色（#RRGGBB）', kind: 'text', mono: true, default: '#FFFFFF' },
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: 'service_url_host（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '破框海报：image-batch-load → 本节点 → breakout-compose（人物来自 matting-birefnet）',
};
