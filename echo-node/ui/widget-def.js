const WIDGET_SPEC = {
  icon: '📣',
  title: '回声',
  fields: [
    { key: 'text', label: 'text（可引用上游输出）', kind: 'text', mono: true, default: '' },
    { key: 'fail', label: 'fail（1=注入失败）', kind: 'select', options: ['0', '1'], default: '0' },
    { key: 'sleep', label: 'sleep 秒', kind: 'slider', min: 0, max: 10, step: 0.5, default: 0 },
    { key: 'marker', label: 'marker 标记文件路径', kind: 'text', mono: true, default: '' },
  ],
  note: '测试节点：marker 行数=执行次数；fail 注入失败；验证重置重跑',
}
