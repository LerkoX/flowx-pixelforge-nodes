const WIDGET_SPEC = {
  icon: '🧩',
  title: '视频拼接',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'videos', label: '视频 ID 列表（逗号分隔，按序拼接）', kind: 'text', mono: true },
    { key: 'crossfade', label: '音频交叉淡化秒数', kind: 'slider', min: 0, max: 2, step: 0.1, default: 0.5 },
    { key: 'filename_prefix', label: '文件名前缀', kind: 'text', default: 'flowx-long' },
    { key: 'output_dir', label: '输出目录', kind: 'text', mono: true, default: '~/flowx-output' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  mediaVideoOutput: 'file_path',
  note: '多段拼接为长视频：视频 concat + 音频 acrossfade 淡化',
}
