const WIDGET_SPEC = {
  icon: '🧠',
  title: 'Checkpoint 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'ckpt_name', label: '模型文件名（ckpt_name）', kind: 'model', modelType: 'checkpoint' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
    { key: 'dtype', label: '计算精度旋钮：auto（dtype）', kind: 'text', default: 'auto' },
    { key: 'offload', label: '显存治理旋钮：auto（offload）', kind: 'text', default: 'auto' },
    { key: 'use_t5', label: 'SD3 T5-XXL 文本编码器开关：auto（use_t5）', kind: 'text', default: 'auto' },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5 },
  ],
  note: '将指定 checkpoint',
}
