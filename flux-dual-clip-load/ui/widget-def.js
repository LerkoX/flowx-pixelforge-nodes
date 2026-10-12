const WIDGET_SPEC = {
  icon: '📎',
  title: 'FLUX DualCLIP 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 't5', label: 'T5-XXL GGUF', kind: 'text', mono: true, default: 't5xxl-Q4_K_S.gguf' },
    { key: 'dtype', label: 'CLIP-L 精度（dtype，T5 恒 bf16）', kind: 'select', options: ['auto', 'fp16', 'bf16', 'fp32'], default: 'auto' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒，异步任务超时）', kind: 'text', default: 1800, advanced: true },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5, advanced: true },
  ],
  note: 'CLIP-L 立即装载；T5 编码相位才上卡（内存错峰）；异步 /jobs 轮询，隧道下也稳',
}
