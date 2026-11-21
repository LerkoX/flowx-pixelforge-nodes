const WIDGET_SPEC = {
  icon: '🖼️',
  title: 'Qwen-Image 生成',
  fields: [
    { key: 'prompt', label: '正向提示词', kind: 'textarea' },
    { key: 'negative_prompt', label: '负向提示词', kind: 'textarea' },
    { key: 'steps', label: '步数（50 / Lightning 4-8）', kind: 'number' },
    { key: 'true_cfg_scale', label: 'True CFG（4.0 / Lightning 1.0）', kind: 'number' },
    { key: 'width', label: '宽（16 倍数）', kind: 'number' },
    { key: 'height', label: '高（16 倍数）', kind: 'number' },
    { key: 'seed', label: '种子（-1=随机）', kind: 'number' },
    { key: 'strength', label: '重绘强度（inpaint）', kind: 'number', advanced: true },
    { key: 'padding_mask_crop', label: '遮罩裁剪外扩（0=关）', kind: 'number', advanced: true },
    { key: 'mask_feather', label: '合成羽化像素', kind: 'number', advanced: true },
    { key: 'max_sequence_length', label: '文本最大 token（0=装配值）', kind: 'number', advanced: true },
    { key: 'num_images', label: '张数', kind: 'number', advanced: true },
  ],
  note: 'Qwen-Image 生成（T2I / 编辑 / 局部重绘——image+mask：edit pipe=合成（推荐），inpaint pipe=管线重绘）',
}
