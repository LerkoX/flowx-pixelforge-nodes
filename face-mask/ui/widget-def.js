const WIDGET_SPEC = {
  icon: '😶',
  title: '人脸蒙版',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'image', label: '输入图像（上游 image 输出）', kind: 'text', mono: true },
    { key: 'face_index', label: '选脸 face_index（-1=最大脸）', kind: 'slider', min: -1, max: 9, step: 1, default: -1 },
    { key: 'det_thresh', label: '检测阈值 det_thresh', kind: 'slider', min: 0.05, max: 0.9, step: 0.05, default: 0.2 },
    { key: 'expand', label: '外扩 expand', kind: 'slider', min: 0, max: 2, step: 0.1, default: 0.6 },
    { key: 'feather', label: '羽化 feather(px)', kind: 'slider', min: 0, max: 64, step: 2, default: 16 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  replay: ['face_index', 'det_thresh', 'expand', 'feather'],
  compareInput: 'image',
  note: '人脸 bbox → 羽化蒙版 + crop 框坐标 → 接 image-crop / image-composite 做 FaceDetailer',
}
