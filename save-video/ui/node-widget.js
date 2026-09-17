/**
 * save-video 画布组件：mp4 播放器 + 保存路径/体积展示。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 * flowx-studio 外壳零改动：渲染权在节点 widget。
 * 视频源优先级：/api/v1/media/file 直读本地文件（无限大小、可拖进度条）
 *   → video_b64 内嵌兜底（远程执行器等文件不在本机的场景）→ 路径文本。
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

// Studio 本地媒体文件端点（同源，浏览器自动携带认证 cookie）
const mediaUrl = (p) => '/api/v1/media/file?path=' + encodeURIComponent(p)

// 视频放大播放 lightbox：大尺寸 <video> + 显式全屏按钮（requestFullscreen，
// iOS Safari 回退 webkitEnterFullscreen；原生控制条也自带全屏入口）。
// startTime 同步自内嵌播放器；onClose 回调负责进度回同步。返回 overlay。
function buildVideoLightbox(src, caption, startTime, onClose) {
  const overlay = h('div', [
    'position:fixed', 'inset:0', 'z-index:9999',
    'background:rgba(0,0,0,0.88)', 'backdrop-filter:blur(4px)',
    'display:flex', 'align-items:center', 'justify-content:center',
  ].join(';'))
  const big = h('video', [
    'max-width:94vw', 'max-height:86vh', 'background:#000',
    'border-radius:8px', 'box-shadow:0 8px 40px rgba(0,0,0,0.6)',
  ].join(';'))
  big.controls = true
  big.loop = true
  big.playsInline = true
  big.src = src
  big.addEventListener('click', (e) => e.stopPropagation())

  const BTN_CSS = [
    'position:absolute', 'width:34px', 'height:34px',
    'border-radius:50%', 'border:1px solid rgba(255,255,255,0.25)',
    'background:rgba(255,255,255,0.10)', 'color:rgba(255,255,255,0.85)',
    'font-size:16px', 'line-height:1', 'cursor:pointer', 'z-index:2',
  ].join(';')
  const closeBtn = h('button', BTN_CSS + ';top:14px;right:16px', '✕')
  closeBtn.title = '关闭（Esc）'
  closeBtn.addEventListener('click', (e) => { e.stopPropagation(); onClose && onClose() })
  const fsBtn = h('button', BTN_CSS + ';top:14px;right:58px', '⛶')
  fsBtn.title = '全屏播放'
  fsBtn.addEventListener('click', (e) => {
    e.stopPropagation()
    if (big.requestFullscreen) big.requestFullscreen().catch(() => {})
    else if (big.webkitEnterFullscreen) big.webkitEnterFullscreen() // iOS Safari
    else if (overlay.requestFullscreen) overlay.requestFullscreen().catch(() => {})
  })
  overlay.append(big, closeBtn, fsBtn)
  if (caption) {
    overlay.append(h('div', [
      'position:absolute', 'left:0', 'right:0', 'bottom:10px', 'text-align:center',
      'font-size:10px', 'color:rgba(255,255,255,0.45)', 'pointer-events:none',
      'padding:0 16px', 'word-break:break-all',
    ].join(';'), caption))
  }
  overlay.addEventListener('click', () => onClose && onClose())
  document.body.appendChild(overlay)
  try { big.currentTime = startTime || 0 } catch (_) {}
  big.play().catch(() => {})
  return { overlay, video: big }
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
  // 放大播放入口（有视频源时显示）
  const expandBtn = h('button', [
    'position:absolute', 'top:6px', 'right:6px', 'z-index:3', 'display:none',
    'background:rgba(0,0,0,0.55)', 'border:1px solid rgba(255,255,255,0.2)',
    'border-radius:6px', 'padding:2px 8px', 'font-size:10px',
    'color:rgba(255,255,255,0.85)', 'cursor:pointer',
  ].join(';'), '⛶ 放大播放')
  playerBox.append(placeholder, videoEl, expandBtn)

  const pathLine = h('div', 'color:rgba(255,255,255,0.45);word-break:break-all;font-family:ui-monospace,Menlo,monospace;font-size:10px')
  const sizeLine = h('div', 'color:rgba(255,255,255,0.35);font-size:10px')

  // 直读本地文件失败时（文件不在 server 白名单/远程执行器产物）回退 base64
  let fallbackB64 = ''
  videoEl.addEventListener('error', () => {
    if (fallbackB64) {
      videoEl.src = 'data:video/mp4;base64,' + fallbackB64
      fallbackB64 = ''
    } else {
      videoEl.style.display = 'none'
      videoEl.removeAttribute('src')
      placeholder.style.display = ''
      placeholder.textContent = '已保存（无法在线播放，见下方路径）'
    }
  })

  // 放大播放 lightbox：同步内嵌进度，关闭时回同步并恢复内嵌播放
  let vbox = null
  const closeVbox = () => {
    if (!vbox) return
    document.removeEventListener('keydown', onVboxKey, true)
    const v = vbox.video
    const wasPlaying = v && !v.paused
    const t = v ? v.currentTime : 0
    vbox.overlay.remove()
    vbox = null
    if (videoEl.getAttribute('src')) {
      try { videoEl.currentTime = t } catch (_) {}
      if (wasPlaying) videoEl.play().catch(() => {})
    }
  }
  const onVboxKey = (e) => {
    if (e.key === 'Escape') {
      e.stopPropagation()
      closeVbox()
    }
  }
  expandBtn.addEventListener('click', (e) => {
    e.stopPropagation()
    const src = videoEl.getAttribute('src')
    if (!src || vbox) return
    const cur = (lastProps && lastProps.outputs) || {}
    const wasPlaying = !videoEl.paused
    vbox = buildVideoLightbox(src, cur.file_path || '', videoEl.currentTime, closeVbox)
    document.addEventListener('keydown', onVboxKey, true)
    if (wasPlaying) videoEl.pause()
  })

  el.append(header, playerBox, pathLine, sizeLine)

  let lastProps = props

  function render(p) {
    lastProps = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const o = p.outputs || {}
    if (o.file_path) {
      // 优先直读本地文件：无内嵌体积上限，支持 Range 拖动进度条
      fallbackB64 = o.video_b64 || ''
      const src = mediaUrl(o.file_path)
      if (videoEl.getAttribute('src') !== src) videoEl.src = src
      videoEl.style.display = 'block'
      placeholder.style.display = 'none'
      expandBtn.style.display = ''
    } else if (o.video_b64) {
      fallbackB64 = ''
      const src = 'data:video/mp4;base64,' + o.video_b64
      if (videoEl.getAttribute('src') !== src) videoEl.src = src
      videoEl.style.display = 'block'
      placeholder.style.display = 'none'
      expandBtn.style.display = ''
    } else {
      videoEl.style.display = 'none'
      videoEl.removeAttribute('src')
      placeholder.style.display = ''
      expandBtn.style.display = 'none'
      placeholder.textContent = p.status === 'success'
        ? '已保存（无播放源，见下方路径）'
        : (p.status === 'running' ? '下载中…' : '等待执行…')
    }
    pathLine.textContent = o.file_path || ''
    sizeLine.textContent = o.size_bytes ? fmtBytes(o.size_bytes) : ''
  }

  render(props)

  return {
    update(next) { render(next) },
    unmount() {
      closeVbox()
      videoEl.pause()
      videoEl.removeAttribute('src')
      el.textContent = ''
    },
  }
}
