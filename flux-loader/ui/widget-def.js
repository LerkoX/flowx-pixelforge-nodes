const WIDGET_SPEC = {
  icon: '⚡',
  title: 'FLUX 加载',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'transformer', label: 'transformer GGUF 文件名', kind: 'text', mono: true, placeholder: 'flux1-schnell-Q4_K_S.gguf' },
    { key: 't5', label: 'T5-XXL GGUF 文件名', kind: 'text', mono: true, placeholder: 't5xxl-Q4_K_S.gguf' },
    { key: 'offload', label: 'offload', kind: 'select', options: ['auto', 'none'], default: 'auto', advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: 'FLUX.1-schnell GGUF Q4 装配（幂等）；输出 model_ref 接 FLUX 编码/采样节点',
}
