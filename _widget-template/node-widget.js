/**
 * __NAME__ 画布组件：执行状态 + 结果缩略图预览（轻量模板，9 个图像/蒙版节点共用）。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 * 预览源：节点执行后 Metadata 里的 thumb_b64（data URL，节点侧 PIL 生成），
 * 不经服务端中转，浏览器无跨域/鉴权问题；点击弹出可缩放 lightbox。
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

// 可缩放图片 lightbox：滚轮缩放、双击 1x↔3x、触屏双指捏合缩放、放大后
// 单指/鼠标拖动平移。点遮罩空白 / ✕ / Esc 关闭；缩放或拖动手势后的抬起
// 不会误触发关闭。（与 load-image widget 同款实现）
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

  let scale = 1, tx = 0, ty = 0
  const pointers = new Map()
  let pinch0 = null
  let gesture = false
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
      const pts = [...pointers.values()]
      const m = midOf(pts[0], pts[1])
      scale = clampScale(pinch0.scale * distOf(pts[0], pts[1]) / pinch0.dist)
      tx = pinch0.tx + (m.x - pinch0.mx)
      ty = pinch0.ty + (m.y - pinch0.my)
      if (scale <= 1) { scale = 1; tx = 0; ty = 0 }
      gesture = true
      apply()
    } else if (pointers.size === 1 && scale > 1) {
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
  title.textContent = '__ICON__ __TITLE__'
  title.style.fontSize = '12px'
  const statusLabel = document.createElement('span')
  statusLabel.style.cssText = 'color:rgba(255,255,255,0.35);margin-left:auto'
  header.append(dot, title, statusLabel)

  // 结果缩略图预览区
  const imgWrap = document.createElement('div')
  imgWrap.style.cssText = [
    'flex:1', 'min-height:60px', 'display:flex', 'align-items:center', 'justify-content:center',
    'background:rgba(0,0,0,0.25)', 'border-radius:8px', 'overflow:hidden', 'margin-bottom:6px',
  ].join(';')
  const placeholder = document.createElement('span')
  placeholder.style.cssText = 'color:rgba(255,255,255,0.25);font-size:10px;padding:8px;text-align:center'
  placeholder.textContent = '执行后显示结果预览'
  const img = document.createElement('img')
  img.style.cssText = 'max-width:100%;max-height:100%;object-fit:contain;display:none'
  img.alt = 'result'
  img.addEventListener('error', () => {
    img.style.display = 'none'
    placeholder.style.display = ''
    placeholder.textContent = '缩略图损坏'
  })
  img.addEventListener('load', () => {
    img.style.display = 'block'
    placeholder.style.display = 'none'
  })
  imgWrap.append(placeholder, img)

  // 点击缩略图弹出可缩放 lightbox
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
    lightbox = buildImageLightbox(img.getAttribute('src'), '', closeLightbox)
    document.addEventListener('keydown', onLightboxKey, true)
  })

  const infoLine = document.createElement('div')
  infoLine.style.cssText = 'color:rgba(255,255,255,0.45);word-break:break-all;font-size:10px;flex-shrink:0'

  el.append(header, imgWrap, infoLine)

  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status

    const o = p.outputs || {}
    const thumb = o.thumb_b64 || ''
    if (thumb) {
      if (img.getAttribute('src') !== thumb) img.src = thumb
      imgWrap.style.cursor = 'zoom-in'
      imgWrap.title = '点击放大查看'
    } else {
      img.removeAttribute('src')
      img.style.display = 'none'
      imgWrap.style.cursor = ''
      imgWrap.title = ''
      placeholder.style.display = ''
      placeholder.textContent = p.status === 'running' ? '处理中…' : '执行后显示结果预览'
    }

    const oid = o.image || o.mask || ''
    const dims = (o.width && o.height) ? ` ${o.width}×${o.height}` : ''
    infoLine.textContent = oid ? `${o.mask ? 'mask' : 'image'}: ${oid}${dims}` : ''
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
