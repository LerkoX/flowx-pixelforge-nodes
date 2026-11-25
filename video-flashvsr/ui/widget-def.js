const WIDGET_SPEC = {
  icon: '⚡',
  title: 'FlashVSR 视频超分',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: '画布可达地址 service_url_host', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '源视频对象 ID（video）', kind: 'text', mono: true },
    { key: 'scale', label: '放大倍率 scale（官方推荐 4.0）', kind: 'slider', min: 1, max: 4, step: 0.5, default: 4.0 },
    { key: 'sparse_ratio', label: '稀疏注意力 sparse_ratio（1.5快/2.0稳）', kind: 'slider', min: 1.0, max: 3.0, step: 0.1, default: 2.0 },
    { key: 'local_range', label: '局部窗口 local_range（9锐/11稳）', kind: 'slider', min: 5, max: 15, step: 2, default: 11 },
    { key: 'seed', label: '随机种子 seed', kind: 'text', default: 0 },
    { key: 'color_fix', label: '颜色校正 color_fix（1开0关）', kind: 'text', default: 1 },
    { key: 'max_frames', label: '最多帧数 max_frames（0=全片）', kind: 'text', default: 0 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '一步蒸馏流式视频超分（4× 推荐）；输出向下取 128 倍数：480×832→1920×3328。需要 block_sparse_attn + 权重五件套',
}
