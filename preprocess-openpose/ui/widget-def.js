const WIDGET_SPEC = {
  icon: '🦴',
  title: 'OpenPose 姿态提取',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'image', label: 'image（人物图）', kind: 'text', mono: true },
    { key: 'include_hand', label: '手部骨架 include_hand', kind: 'check', default: false },
    { key: 'include_face', label: '面部特征 include_face', kind: 'check', default: false },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  replay: ['include_hand', 'include_face'],
  note: '输出骨架图 → controlnet-apply 的 hint（配 openpose ControlNet）',
}
