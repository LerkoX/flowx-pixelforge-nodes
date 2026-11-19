const WIDGET_SPEC = {
  icon: '🎭',
  title: '人物遮罩（SAM3 / SAM2+深度）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '源视频对象 ID', kind: 'text', mono: true },
    { key: 'prompt', label: '目标人物描述（英文更稳）', kind: 'text', default: 'person' },
    { key: 'obj_pick', label: '多目标策略', kind: 'select', options: ['all', 'best'], default: 'all', advanced: true },
    { key: 'dilate', label: '遮罩膨胀像素', kind: 'slider', min: 0, max: 40, step: 1, default: 12 },
    { key: 'shape', label: '遮罩形状', kind: 'select', options: [{ value: 'contour', label: 'contour 精细轮廓（默认）' }, { value: 'bbox', label: 'bbox 包围矩形（Wan replace 官方配方，治轮廓黑影）' }], default: 'contour' },
    { key: 'bbox_smooth', label: 'bbox 防抖窗口（帧）', kind: 'slider', min: 1, max: 15, step: 2, default: 5, advanced: true },
    { key: 'motion_union', label: '帧差运动兜底（防裙摆漏罩）', kind: 'toggle', default: false },
    { key: 'mosaic', label: '马赛克块边（0=关）', kind: 'slider', min: 0, max: 32, step: 2, default: 0 },
    { key: 'max_frames', label: '最多帧数（0=全片）', kind: 'number', default: 0, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'mask_model=sam3 文本分割(1/8) | sam2=SAM3首帧+SAM2追踪(256网格) | depth=深度Otsu二值化(拓扑干净,人物须为最近前景) | rvm=视频抠人(循环记忆,时序最稳,软alpha) | birefnet=单帧SOTA(边缘最锐,人物用portrait版) | depth_refine>0 深度精修(仅sam系)。输出灰度遮罩(rvm/birefnet为软alpha)',
}
