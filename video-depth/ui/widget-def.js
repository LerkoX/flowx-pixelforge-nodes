const WIDGET_SPEC = {
  icon: '🌊',
  title: '视频深度（Depth-Anything-V2）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '源视频对象 ID（接 load-video）', kind: 'text', mono: true },
    { key: 'invert', label: '反转深度极性（开=近暗远亮，对齐 ControlNet depth 惯例；关=近亮远暗，模型原生）', kind: 'toggle', default: false },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'number', default: 1800, advanced: true },
    { key: 'poll_interval', label: '轮询间隔（秒）', kind: 'number', default: 5, advanced: true },
  ],
  note: 'Depth-Anything-V2-Base 逐帧深度（9 帧 ≈2s）。输出深度视频 + 首帧深度图（画布预览）。人物/前景通常明显比背景近，可直接用于前后关系判定或 ControlNet depth 条件',
}
