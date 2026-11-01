const WIDGET_SPEC = {
  icon: '🎯',
  title: '蒙版BBox裁剪',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: '画布可达地址 service_url_host（预览中转用）', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '原图对象 ID（image）', kind: 'text', mono: true },
    { key: 'mask', label: '蒙版对象 ID（白=重绘区）', kind: 'text', mono: true },
    { key: 'padding', label: 'bbox 外扩像素 padding', kind: 'slider', min: 0, max: 256, step: 8, default: 64 },
    { key: 'target_long_side', label: 'crop 长边像素 target_long_side', kind: 'slider', min: 512, max: 2048, step: 64, default: 1536 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '按蒙版白区 bbox+padding 裁剪并等比缩放到长边 target_long_side（8 对齐）；执行后预览帧显示 crop 结果图，节点返回区可见 bbox(x,y,width,height) 与 crop 实际尺寸（crop_width×crop_height）',
}
