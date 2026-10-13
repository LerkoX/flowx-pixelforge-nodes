// catvton-load 的节点 UI 定义（由 _tools/build-widget.py 与 _common/widget-base.js 拼接）
// 只声明本节点特有的控件与读数；画布通用件（状态/预览/输出）由 widget-base 提供。
const WIDGET_SPEC = {
  icon: '👗',
  title: 'CatVTON 装载',
  fields: [
    { key: 'variant', label: '权重变体', kind: 'select',
      options: ['mix', 'vitonhd', 'dresscode'], default: 'mix' },
    { key: 'dtype', label: 'UNet 精度', kind: 'select',
      options: ['fp32', 'fp16'], default: 'fp32' },
    { key: 'vae_dtype', label: 'VAE 精度', kind: 'select',
      options: ['fp32', 'fp16'], default: 'fp32' },
    { key: 'chunk', label: 'attention 分块', kind: 'slider',
      min: 0, max: 4096, step: 512, default: 1024 },
    { key: 'service_url', label: '服务地址', kind: 'text', advanced: true },
    { key: 'service_url_host', label: '服务地址（画布侧）', kind: 'text', advanced: true },
    { key: 'service_token', label: '令牌', kind: 'text', advanced: true },
  ],
  note: 'Pascal(1080) 必需：fp32 + 分块>0（不分块会物化约 19GB 分数矩阵，实测 7 分钟/步都跑不完）',
}
