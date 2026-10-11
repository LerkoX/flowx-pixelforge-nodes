const WIDGET_SPEC = {
  icon: '📦',
  title: 'GroundingDINO 装载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'variant', label: '变体', kind: 'select', options: ['tiny', 'base'], default: "tiny" },
    { key: 'dtype', label: '精度', kind: 'select', options: ['fp32', 'fp16', 'auto'], default: "fp32" },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'tiny 已够用（实测 person/dress/shoe 均能命中）；卸载用 model-unload（键 gd:tiny）',
};
