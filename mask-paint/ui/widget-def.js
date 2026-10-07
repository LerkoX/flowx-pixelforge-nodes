const WIDGET_SPEC = {
  icon: '🖌',
  title: '蒙版手绘',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '原图对象 ID（image）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  paint: { strokesKey: 'strokes_json', imageInput: 'image' },
  note: '点缩略图或「✏️ 编辑蒙版」手绘重绘区域（白=重绘/黑=保留）；执行时按原图实际分辨率光栅化。底图取自上次执行的输入图',
}
