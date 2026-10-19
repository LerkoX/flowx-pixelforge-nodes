const WIDGET_SPEC = {
  icon: '📦',
  title: 'DDColor 上色装载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'variant', label: 'DDColor 变体', kind: 'select', options: ['modelscope', 'paper', 'artistic', 'paper_tiny'], default: "modelscope" },
    { key: 'dtype', label: '推理精度', kind: 'select', options: ['fp32', 'fp16', 'auto'], default: "fp32" },
    { key: 'reload', label: '强制重装', kind: 'select', options: ['false', 'true'], default: "false" },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '变体权重：modelscope 912MB（默认）/ paper 912MB / artistic 912MB / paper_tiny 220MB；装载后常驻 LRU（可被淘汰，也可用 model-unload 卸载）',
};
