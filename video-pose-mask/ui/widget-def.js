const WIDGET_SPEC = {
  icon: '🦴',
  title: '骨架派生遮罩（紧身轮廓）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: 'DWPose 骨架视频对象 ID', kind: 'text', mono: true },
    { key: 'grow', label: '骨架膨胀像素（长出紧身轮廓）', kind: 'slider', min: 0, max: 80, step: 2, default: 30 },
    { key: 'feather', label: '软边高斯核（奇数，1=硬边）', kind: 'slider', min: 1, max: 41, step: 2, default: 9 },
    { key: 'threshold', label: '骨架二值化阈值', kind: 'slider', min: 1, max: 100, step: 1, default: 10, advanced: true },
    { key: 'limit_mask', label: '约束遮罩（原人精确mask，可选）', kind: 'text', mono: true },
    { key: 'limit_expand', label: '约束膨胀像素（裁掉灰翅膀）', kind: 'slider', min: 0, max: 20, step: 1, default: 4 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '流派B 进攻掩码：骨架膨胀成紧身轮廓 → 作为条件暗谷只压暗这块 → 逼模型画瘦人物；原人宽轮廓区保持亮窗帘由模型补背景。grow 30~40 为大改小推荐；feather 9~15 太小会产生硬边台阶、太大则灰雾。',
}
