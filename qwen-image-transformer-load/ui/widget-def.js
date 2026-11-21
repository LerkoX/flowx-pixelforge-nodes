const WIDGET_SPEC = {
  icon: '🧱',
  title: 'Qwen-Image Transformer 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'model', label: '官方仓库目录名', kind: 'text', mono: true, advanced: true },
    { key: 'transformer', label: 'Comfy 单文件（留空=目录）', kind: 'text', mono: true, advanced: true },
    { key: 'dtype', label: '精度（bf16/fp16/fp32）', kind: 'text', mono: true, advanced: false },
    { key: 'offload', label: 'offload（sequential/cuda/none）', kind: 'text', mono: true, advanced: false },
  ],
  note: 'Qwen-Image transformer 加载（20B bf16 逐层 offload）',
}
