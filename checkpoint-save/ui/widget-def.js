const WIDGET_SPEC = {
  icon: '💾',
  title: '存档保存',
  fields: [
    { key: 'media', label: 'media 对象（接上游输出）', kind: 'text', mono: true, default: '' },
    { key: 'name', label: '存档名（全局唯一）', kind: 'text', mono: true, default: '' },
    { key: 'note', label: '备注', kind: 'text', default: '' },
    { key: 'overwrite', label: '同名覆盖', kind: 'select', options: ['0', '1'], default: '0' },
  ],
  note: '钉住对象到命名存档：跨重启/跨执行可回滚，不受对象仓库 TTL/GC 影响',
}
