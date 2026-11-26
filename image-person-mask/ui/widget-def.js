/* global defineNode */
defineNode({
  icon: '🎯',
  title: 'SAM3 图像分割',
  fields: [
    { key: 'image', label: '输入图片', kind: 'model' },
    { key: 'prompt', label: '提示词', kind: 'text' },
    { key: 'obj_pick', label: '目标选择', kind: 'select', options: [
      { value: 'all', label: 'all（并集）' }, { value: 'best', label: 'best（最高分）' }] },
    { key: 'threshold', label: '置信阈值', kind: 'slider', min: 0.1, max: 0.9, step: 0.05 },
    { key: 'dilate', label: '外扩(px)', kind: 'slider', min: 0, max: 40, step: 1 }
  ]
});
