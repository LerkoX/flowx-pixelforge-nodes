const WIDGET_SPEC = {
  icon: '🔧',
  title: 'Qwen-Image 管道组装',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'model', label: '官方仓库目录名', kind: 'text', mono: true, advanced: true },
    { key: 'task', label: '管道类型（t2i/edit/inpaint）', kind: 'text', mono: true, advanced: false },
    { key: 'lora', label: 'LoRA（name@strength 逗号多挂）', kind: 'text', mono: true, advanced: false },
    { key: 'lora_scale', label: 'LoRA 默认强度', kind: 'number', advanced: true },
    { key: 'max_sequence_length', label: '文本最大 token（≤1024）', kind: 'number', advanced: true },
  ],
  note: '组件 → QwenImagePipeline（+ 可选 Lightning LoRA）',
}
