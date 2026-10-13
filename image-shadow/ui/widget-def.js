const WIDGET_SPEC = {
  icon: '🌑',
  title: '环境投影（落地阴影）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '主体图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'mask', label: '主体蒙版对象 ID（留空用 alpha）', kind: 'text', mono: true, advanced: true },
    { key: 'angle', label: '阴影方向（度，90=正下方）', kind: 'slider', min: 0, max: 360, step: 5, default: 90 },
    { key: 'distance', label: '阴影距离(px)', kind: 'slider', min: 0, max: 300, step: 2, default: 24 },
    { key: 'blur', label: '模糊半径(px)', kind: 'slider', min: 0, max: 120, step: 2, default: 24 },
    { key: 'opacity', label: '不透明度', kind: 'slider', min: 0.0, max: 1.0, step: 0.05, default: 0.45 },
    { key: 'squash', label: '压扁（贴地透视，1=不压）', kind: 'slider', min: 0.1, max: 1.0, step: 0.05, default: 0.55 },
    { key: 'fade', label: '远端淡出', kind: 'slider', min: 0.0, max: 1.0, step: 0.05, default: 0.0 },
    { key: 'color', label: '投影颜色', kind: 'text', mono: true, default: '#000000' },
    { key: 'canvas', label: '画布模式', kind: 'select', options: ['expand', 'keep'], default: 'expand' },
    { key: 'background', label: '底图对象 ID（留空输出透明画布）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['angle', 'distance', 'blur', 'opacity', 'squash', 'fade', 'color', 'canvas'],
  compareInput: 'image',
  note: 'shadow_box 之外逐字节不变（可复核）；canvas=expand 主体像素逐字节保留。'
        + '商品图有 alpha 时不必接蒙版节点。',
};
