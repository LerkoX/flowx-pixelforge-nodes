const WIDGET_SPEC = {
  icon: '🎭',
  title: 'Wan 动作预处理',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '原始参考视频对象 ID（接 load-video）', kind: 'text', mono: true },
    { key: 'detector', label: '姿态检测器（dwpose 推荐 / openpose）', kind: 'text', default: 'dwpose' },
    { key: 'stride', label: '抽帧步长（1=每帧）', kind: 'text', default: 1 },
    { key: 'max_frames', label: '最大帧数（0=全部）', kind: 'text', default: 0 },
    { key: 'include_hand', label: '包含手部关键点', kind: 'check', default: false },
    { key: 'include_face', label: '包含面部关键点', kind: 'check', default: false },
    { key: 'pass_through_face', label: 'face_video 用原视频占位', kind: 'check', default: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 3600, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '参考视频 → 骨骼动作视频（DWPose 逐帧 + OpenPose 渲染）',
}
