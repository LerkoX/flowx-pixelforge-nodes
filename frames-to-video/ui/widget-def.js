const WIDGET_SPEC = {
  icon: '🧵',
  title: '帧序列转视频',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: '画布可达地址 service_url_host（docker 用）', kind: 'text', mono: true, advanced: true },
    { key: 'frames', label: '帧序列 IMAGE 批量对象 ID（frames）', kind: 'text', mono: true },
    { key: 'fps', label: '帧率 fps（写入 mp4）', kind: 'slider', min: 1, max: 60, step: 1, default: 24 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '帧序列 → VIDEO（H.264 mp4）；帧尺寸必须一致；落盘用 save-video',
}
