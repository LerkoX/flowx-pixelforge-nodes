const WIDGET_SPEC = {
  icon: '🔍',
  title: '环境变量抓取',
  fields: [
    { key: 'out_file', label: '输出文件', kind: 'text', mono: true, default: '' },
  ],
  note: '调试节点：把 FLOWX_PARAM_*/SERVICE_* 环境变量落盘',
}
