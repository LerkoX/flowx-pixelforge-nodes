const WIDGET_SPEC = {
  icon: '📝',
  title: 'SD3 CLIP 编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'clip_ref', label: 'CLIP 引用 ID（SD3.5 管道视图）（clip_ref）', kind: 'text', mono: true },
    { key: 'text', label: '提示词文本（text）', kind: 'textarea' },
    { key: 'service_url_host', label: '画布可达地址 service_url_host（docker 用）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: '任务超时 job_timeout（秒）', kind: 'text', mono: true, advanced: true },
    { key: 'poll_interval', label: '轮询间隔 poll_interval（秒）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: 'SD3.5 文本编码',
}
