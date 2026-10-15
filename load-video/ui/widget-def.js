const WIDGET_SPEC = {
  icon: '📹',
  title: '加载视频',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: '画布可达地址 service_url_host（docker 预览用）', kind: 'text', mono: true, advanced: true },
    { key: 'video_path', label: '本地视频（可上传 / 拖入）', kind: 'file', accept: 'video/*', mono: true, placeholder: '~/flowx-input/xxx.mp4' },
    { key: 'start', label: '起始帧 start（含）', kind: 'text', default: 0 },
    { key: 'end', label: '结束帧 end（不含，0=到尾）', kind: 'text', default: 0 },
    { key: 'stride', label: '抽帧步长 stride', kind: 'slider', min: 1, max: 12, step: 1, default: 1 },
    { key: 'max_frames', label: '最多帧数 max_frames（0=不限）', kind: 'text', default: 0 },
    { key: 'fps', label: '目标帧率 fps（0=原速）', kind: 'text', default: 0 },
    { key: 'width', label: '宽 width（0=原尺寸）', kind: 'text', default: 0 },
    { key: 'height', label: '高 height（0=原尺寸）', kind: 'text', default: 0 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '本地视频 → 推理服务 VIDEO 对象（不给归一化参数则原字节直通，不重编码）；点「📁 选择文件」或把视频拖进节点即上传到 ~/flowx-input/uploads/ 并写回路径',
}
