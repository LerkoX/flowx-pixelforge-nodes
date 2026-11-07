const WIDGET_SPEC = {
  icon: '🎯',
  title: '九宫格选图编排',
  fields: [
    { label: '底图编排', kind: 'gridpicker', filesKey: 'files', cropsKey: 'crops' },
    { key: 'cell_size', label: '每格边长（px）', kind: 'slider', min: 256, max: 1024, step: 32, default: 640 },
    { key: 'beautify', label: '每格美化（美白/大眼/瘦脸）', kind: 'check', default: false },
    { key: 'whiten', label: '美白强度', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.5, advanced: true },
    { key: 'smooth', label: '磨皮强度', kind: 'slider', min: 0, max: 1, step: 0.05, default: 0.4, advanced: true },
    { key: 'eye', label: '大眼强度', kind: 'slider', min: 0, max: 0.4, step: 0.01, default: 0.12, advanced: true },
    { key: 'slim', label: '瘦脸强度', kind: 'slider', min: 0, max: 0.4, step: 0.01, default: 0.10, advanced: true },
    { key: 'det_thresh', label: '人脸检测阈值', kind: 'slider', min: 0.1, max: 0.9, step: 0.05, default: 0.3, advanced: true },
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'service_url_host', label: 'service_url_host（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '点「✏️ 编排九宫格」弹窗：批量上传 → 点选两格互换 · 格内拖动调裁剪中心 → 完成写回后运行；输出 img1~img9 独立图，拼接交 grid-compose',
};
