const WIDGET_SPEC = {
  icon: '💾',
  title: '保存 3D 网格',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'mesh', label: '推理服务端的 MESH 对象 ID（mesh）', kind: 'text', mono: true },
    { key: 'format', label: '格式提示（glb/obj，GLB 自动识别）（format）', kind: 'text', mono: true, default: 'glb' },
    { key: 'filename_prefix', label: '文件名前缀（自动追加时间戳与后缀）（filename_prefix）', kind: 'text', mono: true, default: 'flowx-mesh' },
    { key: 'output_dir', label: '本地保存目录（支持 ~ 展开）（output_dir）', kind: 'text', mono: true, default: '~/flowx-output' },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '从推理服务下载 3D 网格（GLB/OBJ）保存到本地目录',
}
