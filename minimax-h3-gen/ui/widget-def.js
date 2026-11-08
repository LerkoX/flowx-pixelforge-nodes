const WIDGET_SPEC = {
  icon: '🎞️',
  title: 'MiniMax H3 生成',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'model_ref', label: 'MiniMax H3 MODEL 引用 ID（model_ref）', kind: 'text', mono: true },
    { key: 'prompt', label: '提示词（画面+运动+声音描述）（prompt）', kind: 'text', mono: true },
    { key: 'image', label: '首帧 IMAGE 对象 ID（image，可空）', kind: 'text', mono: true },
    { key: 'last_image', label: '末帧 IMAGE 对象 ID（last_image，可空）', kind: 'text', mono: true },
    { key: 'num_frames', label: '帧数（125 帧 @24fps ≈ 5.2 秒）（num_frames）', kind: 'text', default: 125 },
    { key: 'steps', label: '采样步数（turbo LoRA 用 4）（steps）', kind: 'slider', min: 1, max: 100, step: 1, default: 4 },
    { key: 'seed', label: '随机种子（seed）', kind: 'text', default: -1 },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 7200 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: 'MiniMax H3 联合视频+音频生成（24fps 固定）',
}
