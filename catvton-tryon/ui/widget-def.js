// catvton-tryon 的节点 UI 定义（由 _tools/build-widget.py 与 _common/widget-base.js 拼接）
// 参数来源标注由 widget-base 依据 props.paramSources 渲染（workflow/node/literal 三态）。
const WIDGET_SPEC = {
  icon: '👗',
  title: 'CatVTON 换装',
  fields: [
    { key: 'vton', label: '管道键', kind: 'text', mono: true, advanced: true },
    { key: 'person', label: '模特图', kind: 'text', advanced: true },
    { key: 'cloth', label: '衣服图', kind: 'text', advanced: true },
    { key: 'mask', label: '蒙版（要换掉的旧衣服区）', kind: 'text', advanced: true },
    { key: 'steps', label: '采样步数', kind: 'slider', min: 10, max: 80, step: 1, default: 50 },
    { key: 'guidance', label: 'CFG', kind: 'slider', min: 1, max: 7, step: 0.1, default: 2.5 },
    { key: 'mask_blur', label: '蒙版羽化', kind: 'slider', min: 0, max: 32, step: 1, default: 9 },
    { key: 'repaint', label: '蒙版外保留原图', kind: 'check', default: true },
    { key: 'seed', label: '种子（-1 随机）', kind: 'text' },
    { key: 'width', label: '画布宽', kind: 'text', advanced: true },
    { key: 'height', label: '画布高', kind: 'text', advanced: true },
    { key: 'preview_every', label: '过程帧间隔', kind: 'slider', min: 0, max: 10, step: 1, default: 5, advanced: true },
    { key: 'service_url', label: '服务地址', kind: 'text', advanced: true },
    { key: 'service_url_host', label: '服务地址（画布侧）', kind: 'text', advanced: true },
    { key: 'service_token', label: '令牌', kind: 'text', advanced: true },
  ],
  compareInput: 'person',
  previewEveryKey: 'preview_every',
  note: '专用 VTON 模型：衣服版型/图案来自参考图本身（不是提示词），蒙版没盖到的旧衣服不会被替换',
}
