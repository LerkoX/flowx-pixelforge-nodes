const WIDGET_SPEC = {
  icon: '🎭',
  title: 'FLUX PuLID 人脸编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: 'Studio 宿主机侧（画布/取消兜底）可达的服（service_url_host）', kind: 'text' },
    { key: 'image', label: '参考图对象 ID（image）', kind: 'text', mono: true },
    { key: 'face_index', label: '选哪张脸：-1=最大脸（默认）（face_index）', kind: 'text', default: -1 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600 },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'FLUX PuLID 人脸身份编码',
}
