const WIDGET_SPEC = {
  icon: '🎵',
  title: '加载音频',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'audio_path', label: '本地音频路径（wav/mp3/ogg/m4a）', kind: 'text', mono: true },
    { key: 'service_url_host', label: '容器内回退地址（可空）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  note: '上传为 AUDIO 对象（32kHz 立体声）——H3 Ref2VA 音频参考素材',
}
