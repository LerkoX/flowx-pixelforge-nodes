const WIDGET_SPEC = {
  icon: '🎛️',
  title: '显存策略',
  fields: [
    { key: 'preset', label: '策略', kind: 'select', options: [
      { value: 'saver', label: 'saver · 逐层offload（省显存，实测最快）' },
      { value: 'balanced', label: 'balanced · 块级驻留（实验）' },
      { value: 'turbo', label: 'turbo · 双块级驻留（实验）' }], default: 'saver' },
  ],
  note: '输出 tr/te offload 配置，loader 节点用 {{ VramPolicy.tr_offload }} 引用；一处调参全流水线生效',
};
