const WIDGET_SPEC = {
  icon: '🧭',
  title: 'FLUX ControlNet 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: 'Studio 宿主机侧（画布/取消兜底）可达的服（service_url_host）', kind: 'text' },
    { key: 'cn_path', label: 'ControlNet 目录名（cn_path）', kind: 'text', default: 'cn-union-pro' },
    { key: 'dtype', label: '装载精度：fp16（dtype）', kind: 'text', default: 'fp16' },
    { key: 'quantize', label: '权重量化：int8（quantize）', kind: 'text', default: 'int8' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600 },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5 },
  ],
  note: '装载 FLUX ControlNet',
}
