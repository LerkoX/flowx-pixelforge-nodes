const WIDGET_SPEC = {
  icon: '🎚️',
  title: 'FLUX LoRA 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: '画布侧服务地址（service_url_host）', kind: 'text', mono: true, advanced: true },
    { key: 'model', label: 'FLUX MODEL 对象引用 ID（model）', kind: 'text', mono: true },
    { key: 'lora_name', label: 'LoRA 文件名（lora_name）', kind: 'model', modelType: 'lora' },
    { key: 'strength', label: 'LoRA 强度（strength）', kind: 'text', default: 1.0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600, advanced: true },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5, advanced: true },
  ],
  note: '给 FLUX GGUF transformer 挂 LoRA 补丁（PEFT，可叠加；同名重挂先卸再载）',
}
