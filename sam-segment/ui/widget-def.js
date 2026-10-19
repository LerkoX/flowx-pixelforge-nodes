const WIDGET_SPEC = {
  icon: '🖱️',
  title: 'SAM 点/框选分割',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '输入图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'model', label: 'SAM 模型键', kind: 'text', default: "sam:vit_l", mono: true },
    { key: 'points', label: '点提示（画布点选写入）', kind: 'text', mono: true, advanced: true },
    { key: 'box', label: '框提示（画布框选写入）', kind: 'text', mono: true, advanced: true },
    { key: 'multimask', label: '输出 3 个候选', kind: 'check', default: true },
    { key: 'pick', label: '候选下标（0/1/2）', kind: 'slider', default: 0, min: 0, max: 2, step: 1 },
    { key: 'overlay', label: '蒙版叠加透明度', kind: 'slider', default: 0.5, min: 0, max: 1, step: 0.05 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  pointSelect: { pointsKey: 'points', boxKey: 'box', baseImage: 'image' },
  compareInput: 'image',
  note: '画布点选：点=正样本、Shift/右键点=负样本、拖拽=框选；「▶ 预览」重放分割结果',
};
