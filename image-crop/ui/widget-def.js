const WIDGET_SPEC = {
  icon: '✂️',
  title: '裁剪图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'image', label: '推理服务端的图像对象 ID（图像或蒙版）（image）', kind: 'text', mono: true },
    { key: 'x', label: '左上角 X（像素）（x）', kind: 'text', default: 0 },
    { key: 'y', label: '左上角 Y（像素）（y）', kind: 'text', default: 0 },
    { key: 'width', label: '裁剪宽度（像素）（width）', kind: 'text' },
    { key: 'height', label: '裁剪高度（像素）（height）', kind: 'text' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '矩形裁剪图像（对蒙版同样适用——蒙版即灰度图像）',
}
