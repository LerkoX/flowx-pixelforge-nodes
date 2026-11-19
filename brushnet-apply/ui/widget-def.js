const WIDGET_SPEC = {
  icon: '🧲',
  title: 'BrushNet 捆条件',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'service_url_host', label: '画布侧服务地址（service_url_host）', kind: 'text', mono: true , advanced: true},
    { key: 'control_net', label: 'BrushNet 对象 ID（control_net）', kind: 'text', mono: true },
    { key: 'vae', label: 'VAE 对象 ID（vae）', kind: 'text', mono: true },
    { key: 'image', label: '原图对象 ID（image）', kind: 'text', mono: true },
    { key: 'mask', label: '蒙版对象 ID（白=重绘区）（mask）', kind: 'text', mono: true },
    { key: 'scale', label: '残差强度（scale）', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'start_percent', label: '生效窗口起点（start_percent）', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.0 },
    { key: 'end_percent', label: '生效窗口终点（end_percent）', kind: 'slider', min: 0, max: 1, step: 0.05, default: 1.0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  compareInput: 'image',
  note: 'masked image + mask → 5ch 条件 → ksampler-brushnet 的 control',
}
