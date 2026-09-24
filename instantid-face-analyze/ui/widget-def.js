const WIDGET_SPEC = {
  icon: '🙂',
  title: 'InstantID 人脸分析',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: 'image（参考人脸照）', kind: 'text', mono: true },
    { key: 'face_index', label: '选脸 face_index（-1=最大脸）', kind: 'text', mono: true, default: '-1' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['face_index'],
  note: '输出 face → instantid-apply；kps 关键点图可直接预览核对识别是否正确',
}
