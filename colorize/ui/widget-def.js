const WIDGET_SPEC = {
  icon: '🎨',
  title: 'DDColor 上色',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'model', label: 'DDColor 模型键（model）', kind: 'text', mono: true },
    { key: 'size', label: '内部推理边长(px)', kind: 'slider', default: 512, min: 256, max: 1024, step: 64 },
    { key: 'saturation', label: '色度增强 saturation', kind: 'slider', default: 1.0, min: 0.5, max: 2.5, step: 0.05 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['size', 'saturation'],
  compareInput: 'image',
  note: '保留原图 L 通道只预测 ab ⇒ 亮度/结构不变；chroma 是输出色度均值（灰度输入 chroma_in≈0）。上色是确定性推理（无采样）⇒ 同参数逐字节一致，可作回归判据',
};
