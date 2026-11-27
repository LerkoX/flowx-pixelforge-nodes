const WIDGET_SPEC = {
  icon: '🧹',
  title: '背景抹除（Clean Plate）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '原视频对象 ID', kind: 'text', mono: true },
    { key: 'mask', label: '人物遮罩对象 ID', kind: 'text', mono: true },
    { key: 'dilate', label: '遮罩膨胀像素（覆盖原人所有像素）', kind: 'slider', min: 0, max: 60, step: 1, default: 20 },
    { key: 'radius', label: 'inpaint 半径（telea/ns 用）', kind: 'slider', min: 1, max: 40, step: 1, default: 15 },
    { key: 'method', label: '补全算法', kind: 'select', options: [
    { key: 'mosaic_size', label: '马赛克块宽', kind: 'slider', min: 4, max: 32, step: 2, default: 16 },
        { value: 'lama', label: 'lama 大感受野（推荐，质量最好）' },
        { value: 'ns', label: 'ns Navier-Stokes（大面积较自然）' },
        { value: 'telea', label: 'telea 快（小面积）' }], default: 'lama' },
    { key: 'color_match', label: '色彩对齐（治 LaMa 色偏灰雾）', kind: 'toggle', default: true },
    { key: 'mask_threshold', label: '遮罩阈值（低=捕获半透明边缘）', kind: 'slider', min: 1, max: 127, step: 1, default: 15 },
    { key: 'temporal_union', label: '时序并集（消边缘闪动）', kind: 'toggle', default: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: '任务超时 job_timeout（秒）', kind: 'text', mono: true, advanced: true },
    { key: 'poll_interval', label: '轮询间隔 poll_interval（秒）', kind: 'text', mono: true, advanced: true },
  ],
  note: '流派A 第一阶段：把 mask 区域从视频抹除 → 空无一人的纯净背景（Clean Plate）。lama 需要 big-lama.pt（共享盘已软链 /var/lib/flowx-inference/models/lama/）。dilate 越大抹除越彻底但糊化区越大，建议 20~25。',
}
