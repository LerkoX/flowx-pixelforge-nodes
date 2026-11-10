const WIDGET_SPEC = {
  icon: '📝',
  title: 'H3 文本编码',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'prompt', label: '提示词', kind: 'textarea', rows: 4 },
    { key: 'image', label: '首帧（IMAGE 对象 ID，可选）', kind: 'text', mono: true },
    { key: 'last_image', label: '末帧（IMAGE 对象 ID，可选）', kind: 'text', mono: true },
    { key: 'references', label: 'references JSON（ref2va，可选）', kind: 'textarea', rows: 2, mono: true },
    { key: 'height', label: '高（0=自动，须与 gen 一致）', kind: 'text', default: 0 },
    { key: 'width', label: '宽（0=自动）', kind: 'text', default: 0 },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'text', default: 1800 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'prompt → COND。首末帧原样透传（一条线进、两条线出）→ h3-gen',
}
