const WIDGET_SPEC = {
  icon: '🦴',
  title: '视频控制信号',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '原视频对象 ID（接 load-video）', kind: 'text', mono: true },
    { key: 'mode', label: '模式（depth / pose / canny）', kind: 'text', default: 'depth' },
    { key: 'detector', label: 'pose 检测器（dwpose 推荐）', kind: 'text', default: 'dwpose' },
    { key: 'detect_resolution', label: 'depth 检测分辨率', kind: 'text', default: 512, advanced: true },
    { key: 'include_hand', label: 'pose 含手部', kind: 'check', default: false, advanced: true },
    { key: 'include_face', label: 'pose 含面部', kind: 'check', default: false, advanced: true },
    { key: 'stride', label: '抽帧步长', kind: 'text', default: 1, advanced: true },
    { key: 'max_frames', label: '最大帧数（0=全部）', kind: 'text', default: 0, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '视频 → depth/pose/canny 控制信号视频（给 VACE/Animate 做结构引导）',
}
