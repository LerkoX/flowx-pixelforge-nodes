const WIDGET_SPEC = {
  icon: '🎞️',
  title: 'MiniMax H3 生成',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'model_ref', label: 'MiniMax H3 MODEL 引用 ID（model_ref）', kind: 'text', mono: true },
    { key: 'prompt', label: '提示词（画面+运动+声音描述）（prompt）', kind: 'text', mono: true },
    { key: 'image', label: '首帧 IMAGE 对象 ID（image，可空）', kind: 'text', mono: true },
    { key: 'last_image', label: '末帧 IMAGE 对象 ID（last_image，可空）', kind: 'text', mono: true },
    { key: 'num_frames', label: '帧数（125 帧 @24fps ≈ 5.2 秒）（num_frames）', kind: 'text', default: 117 },
    { key: 'steps', label: '采样步数（turbo-4step LoRA 填 5 = 4 次评估）', kind: 'slider', min: 2, max: 60, step: 1, default: 5 },
    { key: 'seed', label: '随机种子（seed）', kind: 'text', default: -1 },
    { key: 'poll_interval', label: 'poll_interval（秒）', kind: 'text', default: 5 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 7200 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: 'MiniMax H3 联合视频+音频生成（24fps 固定）',
}
