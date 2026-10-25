const WIDGET_SPEC = {
  icon: '🆚',
  title: '前后对比',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image_before', label: '原图对象 ID（image_before）', kind: 'text', mono: true },
    { key: 'image_after', label: '结果图对象 ID（image_after）', kind: 'text', mono: true },
    { key: 'service_url_host', label: '宿主机侧服务地址（service_url_host，可空）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image_before',
  note: '原图 vs 结果滑块对比（拖分割线，点击放大）',
}
