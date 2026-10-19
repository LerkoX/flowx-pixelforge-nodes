const WIDGET_SPEC = {
  icon: '💾',
  title: '批量保存图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'images', label: '对象 ID 列表（JSON）', kind: 'textarea', mono: true },
    { key: 'output_dir', label: '输出目录', kind: 'text', default: "~/flowx-e2e-out", mono: true },
    { key: 'filename_prefix', label: '文件名前缀', kind: 'text', default: "batch" },
    { key: 'start_index', label: '起始序号', kind: 'text', default: "1", mono: true },
    { key: 'zero_pad', label: '序号补零位数', kind: 'slider', default: 3, min: 0, max: 6, step: 1 },
    { key: 'ext', label: '扩展名', kind: 'select', options: ['png', 'jpg'], default: "png" },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '配合多输出算子（如 sam-auto-segment 的 16 个蒙版）把产物全部落盘；文件名 <prefix>_<序号>.png 便于脚本按序复核。只能 local 执行器',
};
