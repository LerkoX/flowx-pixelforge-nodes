const WIDGET_SPEC = {
  icon: '🎲',
  title: '噪点叠加',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'mode', label: '噪点类型', kind: 'select', options: ['gaussian', 'uniform', 'salt_pepper'], default: 'gaussian' },
    { key: 'amount', label: '强度', kind: 'slider', min: 0, max: 1, step: 0.01, default: 0.05 },
    { key: 'monochrome', label: '单色噪点（三通道同噪）', kind: 'check', default: false },
    { key: 'seed', label: '随机种子', kind: 'text', default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
}
