/**
 * save-video 画布组件：mp4 内嵌播放器（video_b64 输出存在时）+ 保存路径/体积展示。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 * flowx-studio 外壳零改动：渲染权在节点 widget。
 */

const STATUS_COLORS = {
  idle: '#94a3b8',
  running: '#22d3ee',
  success: '#34d399',
  failed: '#fb7185',
  skipped: '#64748b',
}

function h(tag, style, text) {
  const n = document.createElement(tag)
  if (style) n.style.cssText = style
  if (text !== undefined) n.textContent = text
  return n
}

const fmtBytes = (n) => {
  n = parseInt(n, 10)
  if (isNaN(n)) return ''
  if (n >= 1 << 20) return (n / (1 << 20)).toFixed(1) + ' MB'
  if (n >= 1 << 10) return (n / (1 << 10)).toFixed(1) + ' KB'
  return n + ' B'
}

export default function mount(el, props) {
  el.style.cssText = [
    'font-family: system-ui, sans-serif',
    'background: rgba(255,255,255,0.04)',
    'border: 1px solid rgba(255,255,255,0.08)',
    'border-radius: 12px',
    'padding: 10px',
    'font-size: 11px',
    'line-height: 1.7',
    'color: rgba(255,255,255,0.85)',
    'overflow: auto',
    'box-sizing: border-box',
    'height: 100%',
  ].join(';')

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '📼 保存视频')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  // 播放器容器：有内嵌视频时 <video>，否则占位框
  const playerBox = h('div', 'position:relative;width:100%;aspect-ratio:16/9;border:1px dashed rgba(255,255,255,0.15);border-radius:8px;background:rgba(0,0,0,0.25);overflow:hidden;margin-bottom:6px;display:flex;align-items:center;justify-content:center;box-sizing:border-box')
  const placeholder = h('span', 'font-size:10px;color:rgba(255,255,255,0.25);padding:8px;text-align:center', '等待执行…')
  const videoEl = h('video', 'display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000')
  videoEl.controls = true
  videoEl.loop = true
  videoEl.muted = true
  videoEl.playsInline = true
  playerBox.append(placeholder, videoEl)

  const pathLine = h('div', 'color:rgba(255,255,255,0.45);word-break:break-all;font-family:ui-monospace,Menlo,monospace;font-size:10px')
  const sizeLine = h('div', 'color:rgba(255,255,255,0.35);font-size:10px')

  el.append(header, playerBox, pathLine, sizeLine)

  function render(p) {
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const o = p.outputs || {}
    if (o.video_b64) {
      const src = 'data:video/mp4;base64,' + o.video_b64
      if (videoEl.getAttribute('src') !== src) videoEl.src = src
      videoEl.style.display = 'block'
      placeholder.style.display = 'none'
    } else {
      videoEl.style.display = 'none'
      videoEl.removeAttribute('src')
      placeholder.style.display = ''
      placeholder.textContent = p.status === 'success'
        ? '已保存（视频超内嵌上限，见下方路径）'
        : (p.status === 'running' ? '下载中…' : '等待执行…')
    }
    pathLine.textContent = o.file_path || ''
    sizeLine.textContent = o.size_bytes ? fmtBytes(o.size_bytes) : ''
  }

  render(props)

  return {
    update(next) { render(next) },
    unmount() {
      videoEl.pause()
      videoEl.removeAttribute('src')
      el.textContent = ''
    },
  }
}
