const WIDGET_SPEC = {
  icon: '🧩',
  title: 'H3 LoRA 挂载（独立）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'model', label: '上游 MODEL（可选锚点）', kind: 'text', mono: true },
    { key: 'lora', label: 'LoRA 列表（逗号分隔，name@强度）', kind: 'text', mono: true, default: '' },
    { key: 'lora_scale', label: '默认强度', kind: 'slider', min: 0.1, max: 1.5, step: 0.05, default: 1.0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '独立 H3 LoRA 节点（与 wan-lora-load 同款交互）。输出配置串 → minimax-h3-transformer-load 的 lora 参数。akatz character-swap 官方配方：h3_character_swap_v1.5.safetensors@1.0 + minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors@1.0（8 步 CFG1）',
}
