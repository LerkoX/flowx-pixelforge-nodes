const WIDGET_SPEC = {
  icon: '📼',
  title: '保存视频',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'video', label: '推理服务端的 VIDEO 对象 ID（video）', kind: 'text', mono: true },
    { key: 'filename_prefix', label: '文件名前缀（自动追加时间戳与 .mp4 后缀）（filename_prefix）', kind: 'text', mono: true, default: 'flowx-video' },
    { key: 'output_dir', label: '本地保存目录（支持 ~ 展开）（output_dir）', kind: 'text', mono: true, default: '~/flowx-output' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  mediaVideoOutput: 'file_path',
  note: '从推理服务下载视频 mp4 保存到本地目录',
}
