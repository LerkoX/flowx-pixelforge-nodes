const WIDGET_SPEC = {
  icon: '🔤',
  title: 'Qwen-Image 文本编码器加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'model', label: '官方仓库目录名', kind: 'text', mono: true, advanced: true },
    { key: 'dtype', label: '精度（建议 bf16）', kind: 'text', mono: true, advanced: false },
    { key: 'offload', label: 'offload（sequential/cuda/none）', kind: 'text', mono: true, advanced: false },
  ],
  note: 'Qwen2.5-VL-7B 文本编码器加载（bf16）',
}
