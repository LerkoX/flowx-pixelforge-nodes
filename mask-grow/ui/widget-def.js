const WIDGET_SPEC = {
  icon: '🔵',
  title: '蒙版扩缩',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'mask', label: '推理服务端的蒙版对象 ID（灰度图像）（mask）', kind: 'text', mono: true },
    { key: 'radius', label: '扩张像素半径（radius）', kind: 'slider', min: -64, max: 64, step: 1, default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  note: '蒙版扩张/收缩',
}
