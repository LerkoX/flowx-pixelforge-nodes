const WIDGET_SPEC = {
  icon: '📦',
  title: 'IC-Light 重打光装载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'variant', label: 'IC-Light 变体', kind: 'select', options: ['fc', 'fbc'], default: "fc" },
    { key: 'dtype', label: '推理精度', kind: 'select', options: ['auto', 'fp32', 'fp16'], default: "auto" },
    { key: 'weights', label: '权重路径覆盖（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'reload', label: '强制重装', kind: 'select', options: ['false', 'true'], default: "false" },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'fc=前景条件（默认，靠 light_dir+prompt 打光）；fbc=前景+背景条件（给背景图按图打光）。权重 1.64GB + SD1.5 基座 ⇒ 常驻约 4.3GB；与 flux 常驻（6.4GB）在 8GB 卡上互斥，单链使用或先 model-unload',
};
