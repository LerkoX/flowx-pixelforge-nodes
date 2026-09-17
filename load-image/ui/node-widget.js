/**
 * load-image 画布组件：本地图片预览 + 上传状态卡。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 * 预览源：params.image_path（字面量）经 /api/v1/media/file 直读本地文件，
 * 无需执行即可预览；wired 绑定（{{ Param.xxx }}）时用 paramSources 的当前值预览。
 */

const STATUS_COLORS = {
  idle: '#94a3b8',
  running: '#22d3ee',
  success: '#34d399',
  failed: '#fb7185',
  skipped: '#64748b',
}

// Studio 本地媒体文件端点（同源，浏览器自动携带认证 cookie）
const mediaUrl = (p) => '/api/v1/media/file?path=' + encodeURIComponent(p)

const isWired = (v) => typeof v === 'string' && v.indexOf('{{') >= 0

function h(tag, style, text) {
  const n = document.createElement(tag)
  if (style) n.style.cssText = style
  if (text !== undefined) n.textContent = text
  return n
}

// 可缩放图片 lightbox：滚轮缩放、双击 1x↔3x、触屏双指捏合缩放、放大后
// 单指/鼠标拖动平移。点遮罩空白 / ✕ / Esc 关闭；缩放或拖动手势后的抬起
// 不会误触发关闭。返回已挂到 document.body 的 overlay 元素。
function buildImageLightbox(src, caption, onClose) {
  const overlay = h('div', [
    'position:fixed', 'inset:0', 'z-index:9999',
    'background:rgba(0,0,0,0.85)', 'backdrop-filter:blur(4px)',
    'display:flex', 'align-items:center', 'justify-content:center',
    'cursor:zoom-out', 'touch-action:none', 'user-select:none', '-webkit-user-select:none',
  ].join(';'))
  const big = h('img', [
    'max-width:92vw', 'max-height:88vh', 'object-fit:contain',
    'border-radius:8px', 'box-shadow:0 8px 40px rgba(0,0,0,0.6)',
    'cursor:default', 'touch-action:none',
    'transform-origin:center', 'will-change:transform',
  ].join(';'))
  big.alt = 'preview'
  big.draggable = false
  big.src = src

  // 缩放/平移状态（Pointer Events 统一处理鼠标与触屏）
  let scale = 1, tx = 0, ty = 0
  const pointers = new Map()
  let pinch0 = null
  let gesture = false // 本触点序列发生过缩放/拖动 → 抑制随后的 click 关闭
  const apply = () => { big.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')' }
  const clampScale = (s) => Math.min(10, Math.max(1, s))
  const distOf = (a, b) => Math.hypot(a.x - b.x, a.y - b.y)
  const midOf = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 })

  overlay.addEventListener('pointerdown', (e) => {
    if (e.target.closest('button')) return
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
    if (pointers.size === 1) gesture = false
    try { overlay.setPointerCapture(e.pointerId) } catch (_) {}
    if (pointers.size === 2) {
      const pts = [...pointers.values()]
      const m = midOf(pts[0], pts[1])
      pinch0 = { dist: Math.max(distOf(pts[0], pts[1]), 1), scale: scale, mx: m.x, my: m.y, tx: tx, ty: ty }
    }
    e.preventDefault()
  })
  overlay.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) return
    const prev = pointers.get(e.pointerId)
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
    if (pointers.size === 2 && pinch0) {
      // 双指捏合：按初始间距比例缩放，随中点平移
      const pts = [...pointers.values()]
      const m = midOf(pts[0], pts[1])
      scale = clampScale(pinch0.scale * distOf(pts[0], pts[1]) / pinch0.dist)
      tx = pinch0.tx + (m.x - pinch0.mx)
      ty = pinch0.ty + (m.y - pinch0.my)
      if (scale <= 1) { scale = 1; tx = 0; ty = 0 }
      gesture = true
      apply()
    } else if (pointers.size === 1 && scale > 1) {
      // 放大后单指/鼠标拖动平移
      const dx = e.clientX - prev.x
      const dy = e.clientY - prev.y
      if (Math.abs(dx) + Math.abs(dy) > 2) gesture = true
      tx += dx
      ty += dy
      apply()
    }
    e.preventDefault()
  })
  const endPointer = (e) => {
    pointers.delete(e.pointerId)
    if (pointers.size < 2) pinch0 = null
  }
  overlay.addEventListener('pointerup', endPointer)
  overlay.addEventListener('pointercancel', endPointer)
  overlay.addEventListener('wheel', (e) => {
    e.preventDefault()
    scale = clampScale(scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15))
    if (scale <= 1) { scale = 1; tx = 0; ty = 0 }
    apply()
  }, { passive: false })
  overlay.addEventListener('dblclick', (e) => {
    e.preventDefault()
    if (scale > 1) { scale = 1; tx = 0; ty = 0 } else { scale = 3 }
    apply()
  })

  const closeBtn = h('button', [
    'position:absolute', 'top:14px', 'right:16px', 'width:34px', 'height:34px',
    'border-radius:50%', 'border:1px solid rgba(255,255,255,0.25)',
    'background:rgba(255,255,255,0.10)', 'color:rgba(255,255,255,0.85)',
    'font-size:16px', 'line-height:1', 'cursor:pointer', 'z-index:2',
  ].join(';'), '✕')
  closeBtn.title = '关闭（Esc）'
  closeBtn.addEventListener('click', (e) => { e.stopPropagation(); onClose && onClose() })
  const hint = h('div', [
    'position:absolute', 'top:16px', 'left:16px', 'font-size:10px',
    'color:rgba(255,255,255,0.4)', 'pointer-events:none',
  ].join(';'), '滚轮 / 双指捏合缩放 · 双击 1x↔3x · 放大后拖动平移')
  overlay.append(big, closeBtn, hint)
  if (caption) {
    overlay.append(h('div', [
      'position:absolute', 'left:0', 'right:0', 'bottom:12px', 'text-align:center',
      'font-size:10px', 'color:rgba(255,255,255,0.45)', 'pointer-events:none',
      'padding:0 16px', 'word-break:break-all',
    ].join(';'), caption))
  }
  // 点图片本身不关闭（便于细看/保存），点遮罩空白才关
  big.addEventListener('click', (e) => e.stopPropagation())
  overlay.addEventListener('click', () => {
    if (gesture) { gesture = false; return }
    onClose && onClose()
  })
  document.body.appendChild(overlay)
  return overlay
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
    'display: flex',
    'flex-direction: column',
  ].join(';')

  let cur = props

  const header = document.createElement('div')
  header.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-shrink:0'
  const dot = document.createElement('span')
  dot.style.cssText = 'width:8px;height:8px;border-radius:50%;flex-shrink:0'
  const title = document.createElement('strong')
  title.textContent = '🖼️ Load Image'
  title.style.fontSize = '12px'
  const statusLabel = document.createElement('span')
  statusLabel.style.cssText = 'color:rgba(255,255,255,0.35);margin-left:auto'
  header.append(dot, title, statusLabel)

  // 本地图片预览区
  const imgWrap = document.createElement('div')
  imgWrap.style.cssText = [
    'flex:1', 'min-height:60px', 'display:flex', 'align-items:center', 'justify-content:center',
    'background:rgba(0,0,0,0.25)', 'border-radius:8px', 'overflow:hidden', 'margin-bottom:6px',
  ].join(';')
  const placeholder = document.createElement('span')
  placeholder.style.cssText = 'color:rgba(255,255,255,0.25);font-size:10px;padding:8px;text-align:center'
  placeholder.textContent = '未设置图片路径'
  const img = document.createElement('img')
  img.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain;display:none'
  img.alt = 'source image'
  img.addEventListener('error', () => {
    img.style.display = 'none'
    placeholder.style.display = ''
    placeholder.textContent = '无法预览（路径不存在或不在媒体白名单内）'
  })
  img.addEventListener('load', () => {
    img.style.display = 'block'
    placeholder.style.display = 'none'
  })
  imgWrap.append(placeholder, img)

  // 点击预览图弹出可缩放 lightbox
  let lightbox = null
  const closeLightbox = () => {
    if (!lightbox) return
    document.removeEventListener('keydown', onLightboxKey, true)
    lightbox.remove()
    lightbox = null
  }
  const onLightboxKey = (e) => {
    if (e.key === 'Escape') {
      e.stopPropagation()
      closeLightbox()
    }
  }
  imgWrap.addEventListener('click', () => {
    if (img.style.display === 'none' || !img.getAttribute('src')) return
    if (lightbox) return
    const raw = (cur.params || {}).image_path || ''
    lightbox = buildImageLightbox(img.getAttribute('src'), raw, closeLightbox)
    document.addEventListener('keydown', onLightboxKey, true)
  })

  const pathLine = document.createElement('div')
  pathLine.style.cssText = 'color:rgba(255,255,255,0.45);word-break:break-all;font-size:10px;flex-shrink:0'
  const imageLine = document.createElement('div')
  imageLine.style.cssText = 'color:rgba(255,255,255,0.45);word-break:break-all;font-size:10px;flex-shrink:0'

  el.append(header, imgWrap, pathLine, imageLine)

  // 解析预览路径：字面量直接用；wired 时取 paramSources 的当前值
  function resolvePath(p) {
    const raw = (p.params || {}).image_path || ''
    if (!raw) return ''
    if (!isWired(raw)) return raw
    const src = (p.paramSources || {}).image_path
    if (src && src.kind === 'workflow' && src.paramValue) return String(src.paramValue)
    if (src && src.kind === 'node' && src.runtimeValue) return String(src.runtimeValue)
    return ''
  }

  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status

  const path = resolvePath(p)
    const raw = (p.params || {}).image_path || ''
    if (path) {
      const src = mediaUrl(path)
      if (img.getAttribute('src') !== src) img.src = src
      imgWrap.style.cursor = 'zoom-in'
      imgWrap.title = '点击放大查看'
    } else {
      img.removeAttribute('src')
      img.style.display = 'none'
      imgWrap.style.cursor = ''
      imgWrap.title = ''
      placeholder.style.display = ''
      placeholder.textContent = raw ? '绑定值待解析，执行后可预览' : '未设置图片路径'
    }

    pathLine.textContent = raw ? `path: ${raw}` : ''
    const o = p.outputs || {}
    imageLine.textContent = o.image ? `image: ${o.image}` : (p.status === 'running' ? '上传中…' : '')
  }

  render(props)

  return {
    update(next) { render(next) },
    unmount() {
      closeLightbox()
      el.textContent = ''
    },
  }
}
