const WIDGET_SPEC = {
  icon: '🎨',
  title: 'Wan LoRA / 打光',
  fields: [
    { key: 'model', label: '上游 MODEL（可选锚点）', kind: 'text', mono: true },
    { key: 'lora', label: 'LoRA（逗号分隔，可带强度 a.safetensors:1.0）', kind: 'textarea', rows: 3 },
    { key: 'lora_scale', label: '默认强度', kind: 'text', default: 1 },
    { key: 'relight_lora', label: '打光 LoRA（人物替换建议必开）', kind: 'text', mono: true },
    { key: 'relight_scale', label: '打光强度', kind: 'text', default: 1 },
    { key: 'service_url', label: 'service_url', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 600 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '输出 lora_config → wan-animate2 的 lora 入口（装配期合并）',
}
