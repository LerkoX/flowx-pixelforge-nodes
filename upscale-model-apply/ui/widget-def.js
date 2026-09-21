const WIDGET_SPEC = {
  icon: '🔍',
  title: '模型放大',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'upscale_model', label: 'upscale_model（放大模型加载输出）', kind: 'text', mono: true },
    { key: 'image', label: 'image（待放大图）', kind: 'text', mono: true },
    { key: 'tile', label: '分块 tile（0=整图）', kind: 'slider', min: 0, max: 1024, step: 64, default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  replay: ['tile'],
  compareInput: 'image',
  note: '完成后显示 原图/结果 前后对比；tile>0 分块防大分辨率爆显存',
}
