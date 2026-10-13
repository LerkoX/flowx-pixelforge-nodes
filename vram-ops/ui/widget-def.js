const WIDGET_SPEC = {
  icon: '🧮',
  title: '显存/租约运维',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'action', label: '动作', kind: 'select',
      options: ['status', 'leases', 'unload', 'interrupt', 'policy_get', 'policy_set'],
      default: 'status' },
    { key: 'target', label: '卸载目标（unload；空=全部）', kind: 'text', mono: true },
    { key: 'lease', label: '租约视角 lease（unload 时指定，如 727）', kind: 'text', mono: true },
    { key: 'job_id', label: '任务 job_id（interrupt；空=全部）', kind: 'text', mono: true },
    { key: 'policy_json', label: '策略 JSON（policy_set）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'status/leases 看一眼 · unload 清常驻 · interrupt 取消任务 · policy_get/set 调淘汰策略',
}
