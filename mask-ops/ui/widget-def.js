const WIDGET_SPEC = {
  icon: '🧮',
  title: '蒙版布尔运算',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: 'Studio 宿主机侧（画布）可达的服务地址（service_url_host）', kind: 'text' },
    { key: 'mask_a', label: '蒙版 A 对象 ID（mask_a）', kind: 'text' },
    { key: 'mask_b', label: '蒙版 B 对象 ID（灰度图）（mask_b）', kind: 'text' },
    { key: 'op', label: '运算类型：union（并集）/ subtract（op）', kind: 'text', default: 'subtract' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '两张灰度蒙版的布尔运算：union（并集）/ s',
}
