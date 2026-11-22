const WIDGET_SPEC = {
  icon: '🔍',
  title: '脸部追踪裁切（FaceFix）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'video', label: '源视频对象 ID', kind: 'text', mono: true },
    { key: 'mask', label: '脸部遮罩视频对象 ID（person-mask prompt 用 face）', kind: 'text', mono: true },
    { key: 'crop_factor', label: '裁切倍数（脸高 ×N，社区默认 2.5）', kind: 'slider', min: 1.2, max: 4, step: 0.1, default: 2.5 },
    { key: 'out_size', label: '裁片输出边长（px）', kind: 'select', options: ['512', '768', '1024'], default: '768' },
    { key: 'ema', label: '包围盒 EMA 防抖（越大越稳）', kind: 'slider', min: 0, max: 0.95, step: 0.05, default: 0.75 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'job_timeout', label: 'job_timeout（秒）', kind: 'number', default: 1800, advanced: true },
  ],
  note: 'PixelAI H3 FaceFix ①：遮罩逐帧出脸部包围盒 → EMA 平滑 → 方形裁切放大。输出：crop 特写视频（接 H3Gen.inpaint_video）+ mask_white 全白遮罩（接 inpaint_mask）+ boxes（接 face-stitch）+ face_ref 参考图（可作 ref2va 参考）',
}
