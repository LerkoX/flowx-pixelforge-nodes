const WIDGET_SPEC = {
  icon: '🔒',
  title: '脸部光流锁定（Flow Lock）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'base', label: '源视频对象 ID（stitch 的底图）', kind: 'text', mono: true },
    { key: 'refined', label: 'stitch 输出视频对象 ID', kind: 'text', mono: true },
    { key: 'sigma_frac', label: '光流平滑（帧高比例 1/150）', kind: 'slider', min: 0.001, max: 0.02, step: 0.0005, default: 0.0067 },
    { key: 'threshold', label: '改动像素阈值（0-255）', kind: 'slider', min: 0.5, max: 32, step: 0.5, default: 4 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'number', default: 1800, advanced: true },
  ],
  note: 'FaceRefine-Plus 抗摆动：DIS 光流把重绘的脸逐帧锁定回源视频五官位置，治"脸在头上跳舞"。接在 face-stitch 之后；未改动区域直通。官方实测摆动 0.73%→0.19%',
}
