const WIDGET_SPEC = {
  icon: '💾',
  title: '保存图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'image', label: '推理服务端的图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'filename_prefix', label: '文件名前缀（自动追加时间戳与 .png 后缀）（filename_prefix）', kind: 'text', mono: true, default: 'flowx' },
    { key: 'output_dir', label: '本地保存目录（支持 ~ 展开）（output_dir）', kind: 'text', mono: true, default: '~/flowx-output' },
    { key: 'index', label: 'batch>1 时要下载的图像序号（index）', kind: 'text', default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  mediaImageOutput: 'file_path',
  note: '从推理服务下载图像保存到本地（成图直读展示，点击放大）',
}
