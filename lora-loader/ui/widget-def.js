const WIDGET_SPEC = {
  icon: '🎨',
  title: 'LoRA 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'model_ref', label: '底模 MODEL 对象引用 ID（model_ref）', kind: 'text', mono: true },
    { key: 'lora_name', label: 'LoRA 文件名（lora_name）', kind: 'model', modelType: 'lora' },
    { key: 'strength', label: 'LoRA 强度（strength）', kind: 'text', default: 1.0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5 },
  ],
  note: '给 MODEL 挂 LoRA 增量补丁',
}
