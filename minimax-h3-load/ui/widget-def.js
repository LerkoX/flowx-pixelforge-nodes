const WIDGET_SPEC = {
  icon: '🐚',
  title: 'MiniMax H3 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'transformer', label: 'transformer int8 文件名', kind: 'text', mono: true, default: 'minimax_h3_fl2va_int8_convrot.safetensors' },
    { key: 'text_encoder', label: 'Qwen3VL 文本塔 int8 文件名', kind: 'text', mono: true, default: 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors' },
    { key: 'video_vae', label: '视频 VAE int8 文件名', kind: 'text', mono: true, default: 'minimax_h3_video_vae_int8_convrot.safetensors' },
    { key: 'audio_vae', label: '音频 VAE 官方文件名', kind: 'text', mono: true, default: 'minimax_h3_audio_vae_official.safetensors' },
    { key: 'lora', label: 'turbo LoRA 文件名（留空不挂）', kind: 'text', mono: true, default: 'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors' },
    { key: 'lora_scale', label: 'LoRA 强度', kind: 'slider', min: 0, max: 2, step: 0.05, default: 1.0 },
    { key: 'offload', label: 'offload 策略（sequential/none）', kind: 'text', default: 'sequential' },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: 'MiniMax H3 模型装配（int8 反量化 + 逐层 offload）',
}
