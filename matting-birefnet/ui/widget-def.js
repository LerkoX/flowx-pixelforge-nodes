const WIDGET_SPEC = {
  icon: '🪄',
  title: 'BiRefNet 抠图',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'variant', label: '模型变体', kind: 'select', options: ['lite', 'main', 'portrait'], default: 'lite' },
    { key: 'resolution', label: '推理分辨率', kind: 'slider', min: 1024, max: 2048, step: 256, default: 1024 },
    { key: 'refine', label: '去白边（前后景分离）', kind: 'check', default: true },
    { key: 'refine_radius', label: '分离半径', kind: 'slider', min: 8, max: 200, step: 8, default: 90 },
    { key: 'alpha_gamma', label: 'alpha 曲线', kind: 'slider', min: 0.3, max: 3, step: 0.05, default: 1.0 },
    { key: 'use_checkpoint', label: '梯度检查点（省显存）', kind: 'check', default: false },
    { key: 'invert', label: '蒙版反相', kind: 'check', default: false },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  compareInput: 'image',
  note: '输出 mask（灰度 alpha，喂 mask-feather / image-composite / latent-set-noise-mask）+ RGBA 去背图',
};
