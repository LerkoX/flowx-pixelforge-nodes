const WIDGET_SPEC = {
  icon: '🎞️',
  title: 'Motion 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'ckpt_name', label: 'SD1.x 底模名（ckpt_name）', kind: 'text' },
    { key: 'motion_name', label: '运动模块名（motion_name）', kind: 'model', modelType: 'motion' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5 },
  ],
  note: '把 AnimateDiff 运动模块',
}
