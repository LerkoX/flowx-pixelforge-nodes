const WIDGET_SPEC = {
  icon: '🖌',
  title: '蒙版手绘',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image_path', label: '本地底图（可上传 / 拖入）', kind: 'file', accept: 'image/*', mono: true, placeholder: '~/flowx-input/xxx.png', noPreview: true },
    { key: 'image', label: '原图对象 ID（image，有上游时用）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  paint: { strokesKey: 'strokes_json', strokesRefKey: 'strokes_ref', imageInput: 'image', imagePathKey: 'image_path' },
  note: '点缩略图或「✏️ 编辑蒙版」手绘重绘区域（白=重绘/黑=保留）；执行时按原图实际分辨率光栅化。'
    + '笔画保存时自动抽稀并落盘为本地 JSON（参数只留短路径 strokes_ref，旧的内联 strokes_json 仍兼容）。'
    + '底图优先取上游 image；没接上游时直读 image_path（可「📁 选择文件」上传或拖入，需在 Studio 媒体目录白名单内）；'
    + '底图只在上面蒙版区展示（本字段不再重复贴预览图）',
}
