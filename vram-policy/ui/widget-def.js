const WIDGET_SPEC = {
  icon: '🎛️',
  title: '显存策略',
  fields: [
    { key: 'preset', label: '预设', kind: 'select',
      options: ['turbo', 'balanced', 'saver', 'custom'], default: 'turbo' },
    { key: 'tr_offload', label: 'transformer 策略（custom）', kind: 'select',
      options: ['group', 'sequential', 'cuda', 'none'], default: 'group' },
    { key: 'tr_group_blocks', label: 'tr 驻留块数（custom）', kind: 'slider',
      min: 1, max: 16, step: 1, default: 4 },
    { key: 'te_offload', label: '文本塔策略（custom）', kind: 'select',
      options: ['group', 'sequential', 'cuda', 'none'], default: 'group' },
    { key: 'te_group_blocks', label: 'te 驻留块数（custom）', kind: 'slider',
      min: 1, max: 8, step: 1, default: 2 },
  ],
  note: '纯数据节点：策略输出给下游 loader 引用，一处调参全流水线生效。turbo≈12GB 最快；balanced≈7GB；saver≈3GB 最省。',
}
