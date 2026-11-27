const WIDGET_SPEC = {
  icon: '👤',
  title: '人脸裁剪',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像/帧序列（IMAGE 批次逐帧处理）', kind: 'text', mono: true },
    { key: 'pad', label: '外扩比例 pad（方形框）', kind: 'slider', min: 0, max: 2, step: 0.1, default: 0.4 },
    { key: 'out_size', label: '输出边长 out_size', kind: 'text', default: 512 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
  note: '逐帧检测最大人脸 → 方形外扩 → out_size² 裁剪（Wan-Animate face_video 契约）',
}
