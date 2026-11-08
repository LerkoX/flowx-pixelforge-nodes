const WIDGET_SPEC = {
  icon: '📝',
  title: 'H3 文本塔',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'text_encoder', label: 'Qwen3VL int8 文件名', kind: 'text', mono: true, default: 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors' },
    { key: 'dtype', label: '精度（bf16/fp16/fp32）', kind: 'text', default: 'bf16' },
    { key: 'offload', label: 'offload（sequential/none）', kind: 'text', default: 'sequential' },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '文本塔组件（Qwen3VL-32B + tokenizer/processor）→ h3-pipe',
}
