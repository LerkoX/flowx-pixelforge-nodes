const WIDGET_SPEC = {
  icon: '📂',
  title: '批量加载图像',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: '画布可达 service_url_host（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'dir', label: '目录', kind: 'text', default: "~/flowx-input", mono: true },
    { key: 'pattern', label: '文件名通配（; 分隔）', kind: 'text', default: "*.png", mono: true },
    { key: 'sort', label: '排序', kind: 'select', options: ['name', 'mtime', 'size'], default: "name" },
    { key: 'limit', label: '最多取几张（0=全部）', kind: 'slider', default: 0, min: 0, max: 200, step: 1 },
    { key: 'offset', label: '跳过前 N 张', kind: 'slider', default: 0, min: 0, max: 200, step: 1 },
    { key: 'recursive', label: '递归子目录', kind: 'check', default: false },
    { key: 'reverse', label: '倒序', kind: 'check', default: false },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['dir', 'pattern', 'sort', 'limit', 'offset'],
  note: 'Studio 无循环概念 ⇒ 批量只能落在 I/O 上：本节点把目录里的图上传成 ID 列表，给 image-montage（拼联系表）或 image-batch-save（另存目录）用。只能 local 执行器',
};
