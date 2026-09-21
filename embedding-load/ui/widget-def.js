const WIDGET_SPEC = {
  icon: '📚',
  title: 'Embedding 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'clip', label: 'clip（CLIP 对象）', kind: 'text', mono: true },
    { key: 'names', label: 'embedding 名（逗号分隔）', kind: 'text', mono: true, placeholder: 'badhandv4,EasyNegative' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '加载后提示词直接写该词生效（常用于负面压制缺陷）；幂等',
}
