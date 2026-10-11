const WIDGET_SPEC = {
  icon: '🧽',
  title: '蒙版精修（alpha）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'mask', label: '输入蒙版对象 ID（mask）', kind: 'text', mono: true },
    { key: 'threshold', label: '二值化阈值（0=不二值化）', kind: 'slider', min: 0, max: 255, step: 1, default: 0 },
    { key: 'gamma', label: 'alpha 伽马', kind: 'slider', min: 0.3, max: 3, step: 0.05, default: 1.0 },
    { key: 'contrast', label: 'alpha 对比度', kind: 'slider', min: 0.3, max: 3, step: 0.05, default: 1.0 },
    { key: 'feather', label: '羽化半径（px）', kind: 'slider', min: 0, max: 64, step: 1, default: 0 },
    { key: 'grow', label: '扩张/收缩（px）', kind: 'slider', min: -32, max: 32, step: 1, default: 0 },
    { key: 'despeckle', label: '去小杂点（核尺寸，0=关）', kind: 'slider', min: 0, max: 9, step: 2, default: 0 },
    { key: 'invert', label: '蒙版反相', kind: 'check', default: false },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '单一操作请用 mask-grow / mask-feather / mask-invert；本节点用于串联调参',
};
