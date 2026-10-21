const WIDGET_SPEC = {
  icon: '🩺',
  title: '推理服务健康检查',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'expect_resident', label: '【必填】本流水线将使用的模型 expect_resident（逗号分隔常驻键：checkpoint 裸名 / flux-unet:文件 / flux-clip:t5=文件 / flux-vae:vae / cn:名字；清场只卸清单外模型，留空=全卸）', kind: 'text', mono: true },
    { key: 'wait_seconds', label: '服务未就绪时的最长等待秒数（wait_seconds）', kind: 'text', default: 60 },
    { key: 'min_vram_mb', label: '显存闸门 min_vram_mb（0=不检查）', kind: 'text', default: 0, advanced: true },
    { key: 'require_headroom_mb', label: '物理余量闸门 require_headroom_mb（0=不检查）', kind: 'text', default: 0, advanced: true },
    { key: 'evict_unused', label: '清场 evict_unused（淘汰 expect_resident 之外的常驻模型）', kind: 'check', default: true, advanced: true },
    { key: 'queue_wait_seconds', label: 'FIFO 排队最长等待秒数 queue_wait_seconds（0=不排队）', kind: 'text', default: 3600, advanced: true },
    { key: 'queue_poll_seconds', label: '排队轮询间隔 queue_poll_seconds', kind: 'text', default: 5, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: '推理服务闸门：FIFO 排队 + 清场只卸 expect_resident 之外的常驻模型；expect_resident 全命中时跳过 min_vram_mb',
}
