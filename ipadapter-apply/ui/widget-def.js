const WIDGET_SPEC = {
  icon: '🎨',
  title: 'IPAdapter 应用',
  fields: [
    {"key": "service_url", "label": "推理服务 service_url", "kind": "text", "mono": true, "placeholder": "http://…:8100", "advanced": true},
    {"key": "model", "label": "model（底模）", "kind": "text", "mono": true},
    {"key": "ipadapter", "label": "ipadapter（权重）", "kind": "text", "mono": true},
    {"key": "clip_vision", "label": "clip_vision（编码器）", "kind": "text", "mono": true},
    {"key": "image", "label": "image（参考图）", "kind": "text", "mono": true},
    {"key": "weight", "label": "权重 weight", "kind": "slider", "min": 0, "max": 2, "step": 0.05, "default": 0.8},
    {"key": "start_percent", "label": "起始步 start%", "kind": "slider", "min": 0, "max": 1, "step": 0.05, "default": 0},
    {"key": "end_percent", "label": "结束步 end%", "kind": "slider", "min": 0, "max": 1, "step": 0.05, "default": 1},
    {"key": "service_token", "label": "service_token（可空）", "kind": "text", "mono": true, "advanced": true},
  ],
  note: 'weight 0.5~1.0 常用；风格参考可只注前半段（end 0.5）保构图自由。输出 model → ksampler',
}
