const WIDGET_SPEC = {
  icon: '✂️',
  title: '视频转帧序列',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: '画布可达地址 service_url_host（docker 预览用）', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: 'VIDEO 对象 ID（video）', kind: 'text', mono: true },
    { key: 'start', label: '起始帧 start（含）', kind: 'text', default: 0 },
    { key: 'end', label: '结束帧 end（不含，0=到尾）', kind: 'text', default: 0 },
    { key: 'stride', label: '抽帧步长 stride', kind: 'slider', min: 1, max: 12, step: 1, default: 1 },
    { key: 'max_frames', label: '最多帧数 max_frames（0=不限）', kind: 'text', default: 0 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'VIDEO → 帧序列（IMAGE 批量对象）；画布预览显示第 0 帧。帧回视频用 frames-to-video',
}
