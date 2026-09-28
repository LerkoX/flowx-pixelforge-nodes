const WIDGET_SPEC = {
  icon: '🧬',
  title: '人脸相似度',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'image_a', label: '图像 A（基准，上游 image 输出）', kind: 'text', mono: true },
    { key: 'image_b', label: '图像 B（待验证，上游 image 输出）', kind: 'text', mono: true },
    { key: 'face_index_a', label: 'A 选脸 face_index_a（-1=最大脸）', kind: 'slider', min: -1, max: 9, step: 1, default: -1 },
    { key: 'face_index_b', label: 'B 选脸 face_index_b（-1=最大脸）', kind: 'slider', min: -1, max: 9, step: 1, default: -1 },
    { key: 'det_thresh', label: '检测阈值 det_thresh', kind: 'slider', min: 0.05, max: 0.9, step: 0.05, default: 0.5 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  replay: ['face_index_a', 'face_index_b', 'det_thresh'],
  note: '两图目标人脸身份余弦相似度：同人典型 ≥0.5，陌生人 <0.3 —— InstantID 验收量化',
}
