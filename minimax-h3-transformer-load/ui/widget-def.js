const WIDGET_SPEC = {
  icon: '🧠',
  title: 'H3 Transformer',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'transformer', label: 'transformer int8 文件名（ref2va=参考模式）', kind: 'text', mono: true, default: 'minimax_h3_fl2va_int8_convrot.safetensors' },
    { key: 'lora', label: 'turbo LoRA 文件名（留空不挂）', kind: 'text', mono: true, default: 'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors' },
    { key: 'lora_scale', label: 'LoRA 强度', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'dtype', label: '精度（bf16/fp16/fp32）', kind: 'text', default: 'bf16' },
    { key: 'offload', label: 'offload（sequential/none）', kind: 'text', default: 'sequential' },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'transformer 组件加载（int8 反量化 + LoRA + 逐层 offload）→ h3-pipe',
}
