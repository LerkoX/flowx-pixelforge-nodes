const WIDGET_SPEC = {
  icon: '⬜',
  title: 'FLUX 空 Latent',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'width', label: '宽度', kind: 'select', options: ['512', '768', '1024'], default: '768' },
    { key: 'height', label: '高度', kind: 'select', options: ['512', '768', '1024', '1280'], default: '768' },
    { key: 'batch_size', label: '批次', kind: 'text', default: 1 },
    { key: 'job_timeout', label: '任务超时 job_timeout（秒）', kind: 'text', mono: true, advanced: true },
    { key: 'poll_interval', label: '轮询间隔 poll_interval（秒）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '任务式调用（排队不占隧道、可取消）；16 倍数；噪声由采样器按 seed 生成；GTX1080 建议 512~768',
}
