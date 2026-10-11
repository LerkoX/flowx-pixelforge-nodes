const WIDGET_SPEC = {
  icon: '📦',
  title: 'SAM 装载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'variant', label: 'SAM 变体', kind: 'select', options: ['vit_l', 'vit_h', 'vit_b'], default: "vit_l" },
    { key: 'dtype', label: '精度', kind: 'select', options: ['fp32', 'fp16', 'auto'], default: "fp32" },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '装载很贵（实测 vit_l 60~220s、vit_h 163s，取决于权重读盘缓存）⇒ 建议常驻，并优先用 vit_l；卸载用 model-unload 节点（键名 sam:vit_l）',
};
