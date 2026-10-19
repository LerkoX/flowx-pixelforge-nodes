const WIDGET_SPEC = {
  icon: '🩺',
  title: '推理服务健康检查',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'wait_seconds', label: '服务未就绪时的最长等待秒数（wait_seconds）', kind: 'text', default: 60 },
    { key: 'min_vram_mb', label: '显存闸门 min_vram_mb（0=不检查）', kind: 'text', default: 0, advanced: true },
    { key: 'require_headroom_mb', label: '物理余量闸门 require_headroom_mb（0=不检查）', kind: 'text', default: 0, advanced: true },
    { key: 'expect_resident', label: '期望复用模型 expect_resident（逗号分隔，全部常驻则跳过 min_vram_mb）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '检查文生图推理服务；expect_resident 命中常驻缓存时不再为加载预留显存',
}
