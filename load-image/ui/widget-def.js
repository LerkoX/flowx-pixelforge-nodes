const WIDGET_SPEC = {
  icon: '🖼️',
  title: '加载图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'image_path', label: 'FlowX 所在机器的本地图片路径（image_path）', kind: 'text', mono: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  mediaImageParam: 'image_path',
  note: '读取 FlowX 侧本地图片并上传到推理服务（源图直读预览，点击放大）',
}
