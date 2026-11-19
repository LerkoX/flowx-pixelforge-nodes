const WIDGET_SPEC = {
  icon: '🔤',
  title: 'CLIP 编码（Task）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true , advanced: true},
    { key: 'service_url_host', label: '画布侧服务地址（service_url_host）', kind: 'text', mono: true , advanced: true},
    { key: 'clip_ref', label: 'CLIP 模型引用 ID（clip_ref）', kind: 'text', mono: true },
    { key: 'text', label: '提示词（task token 自动追加）', kind: 'textarea', default: '' },
    { key: 'task', label: '任务类型（task）', kind: 'select', options: ['none', 'object_removal', 'object_insertion', 'outpainting', 'shape_guided'], default: 'none' },
    { key: 'side', label: 'cond 侧（side）', kind: 'select', options: ['positive', 'negative'], default: 'positive' },
    { key: 'clip_skip', label: 'CLIP skip（0=不跳）（clip_skip）', kind: 'text', default: 0 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 , advanced: true},
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true , advanced: true},
  ],
  note: 'PowerPaint task prompt 编码（P_ctxt/P_shape/P_obj）',
}
