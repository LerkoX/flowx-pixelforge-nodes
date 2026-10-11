const WIDGET_SPEC = {
  icon: '🖼️',
  title: '加载图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image_path', label: '本地图片（可上传 / 拖入）', kind: 'file', accept: 'image/*', mono: true, placeholder: '~/flowx-input/xxx.png' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '读取 FlowX 侧本地图片并上传到推理服务。点「📁 选择文件」或把图片拖进节点即上传到 ~/flowx-input/uploads/ 并写回路径；也可手工填本机路径',
}
