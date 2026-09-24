const WIDGET_SPEC = {
  icon: '🪪',
  title: 'InstantID 应用',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'model', label: 'model（SDXL checkpoint-loader）', kind: 'text', mono: true },
    { key: 'ipadapter', label: 'ipadapter（instantid-ip-adapter）', kind: 'text', mono: true },
    { key: 'controlnet', label: 'controlnet（instantid-controlnet）', kind: 'text', mono: true },
    { key: 'face', label: 'face（人脸分析输出）', kind: 'text', mono: true },
    { key: 'weight', label: '身份强度 weight', kind: 'slider', min: 0, max: 2, step: 0.05, default: 0.8 },
    { key: 'cn_strength', label: '关键点控制 cn_strength', kind: 'slider', min: 0, max: 2, step: 0.05, default: 0.8 },
    { key: 'start_percent', label: 'IPA 生效起点', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.0, advanced: true },
    { key: 'end_percent', label: 'IPA 生效终点', kind: 'slider', min: 0, max: 1, step: 0.05, default: 1.0, advanced: true },
    { key: 'cn_start_percent', label: 'CN 生效起点', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.0, advanced: true },
    { key: 'cn_end_percent', label: 'CN 生效终点', kind: 'slider', min: 0, max: 1, step: 0.05, default: 1.0, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'SDXL 专用；输出 model+control 接采样节点同名端口；weight=0 & cn_strength=0 ≡ 无 InstantID',
}
