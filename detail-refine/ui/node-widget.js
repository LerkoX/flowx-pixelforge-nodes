/**
 * detail-refine 画布组件：重绘参数卡 + 输出图在线预览。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 * 预览源：outputs.image（推理服务图像对象 ID）经 ${service_url}/images/{id}?thumb=320 直读，
 * <img> 标签跨域不受 CORS 限制；service_url 为 wired 绑定时用 paramSources 的运行时值解析。
 */

const STATUS_COLORS = {
  idle: '#94a3b8',
  running: '#22d3ee',
  success: '#34d399',
  failed: '#fb7185',
  skipped: '#64748b',
}

const METHODS = ['lanczos', 'bicubic', 'bilinear', 'nearest']
const DETECTORS = ['face', 'hand']
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
  const refreshers = []
  const editable = () => typeof cur.onParamsChange === 'function'
  const setParam = (key, value) => {
    if (!editable()) return
    const params = { ...(cur.params || {}), [key]: String(value) }
    cur = { ...cur, params }
    cur.onParamsChange(params)
  }

  // wired 参数（{{ ... }} 绑定）控件：来源标签 + 可编辑文本框。
  const WIRED_INPUT_CSS = 'width:100%;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:6px;padding:4px 7px;font-size:10px;color:rgba(165,180,252,0.95);outline:none;box-sizing:border-box;font-family:ui-monospace,Menlo,monospace'
  const INPUT_CSS = 'width:100%;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:6px;padding:4px 7px;font-size:11px;color:rgba(255,255,255,0.9);outline:none;box-sizing:border-box'
  const LABEL_CSS = 'display:block;font-size:10px;color:rgba(255,255,255,0.45);margin-bottom:2px'
  const sourceCaptionOf = (key, raw) => {
    const src = (cur.paramSources || {})[key]
    if (src && src.kind === 'workflow') {
      return '⚡ 流水线参数 · ' + (src.paramName || '') + (src.paramValue !== undefined ? ' = ' + src.paramValue : '（未定义）')
    }
    if (src && src.kind === 'node') {
      return '🔗 ' + (src.nodeName || src.nodeId || '') + ' · ' + (src.field || '') + (src.runtimeValue !== undefined ? ' = ' + src.runtimeValue : '')
    }
    if (src && src.kind === 'literal') return '✏️ 自定义值'
    return '⟵ ' + (raw !== undefined ? raw : '')
  }
  const sourceCaptionColor = (key) => {
    const src = (cur.paramSources || {})[key]
    if (src && src.kind === 'workflow') return 'rgba(251,191,36,0.85)'
    if (src && src.kind === 'node') return 'rgba(34,211,238,0.8)'
    return 'rgba(165,180,252,0.75)'
  }
  const wiredControl = (key, raw) => {
    const wrap = h('div', '')
    const caption = h('div', 'font-size:9px;margin-bottom:2px;word-break:break-all;line-height:1.4')
    const inp = h('input', WIRED_INPUT_CSS)
    inp.type = 'text'
    inp.value = raw !== undefined ? raw : ''
    inp.spellcheck = false
    inp.title = '参数绑定：输入普通值解除绑定；输入 {{ Param.xxx }} 或 {{ 节点.字段 }} 重新绑定'
    inp.addEventListener('change', () => setParam(key, inp.value.trim()))
    const sync = () => {
      caption.textContent = sourceCaptionOf(key, (cur.params || {})[key])
      caption.style.color = sourceCaptionColor(key)
      inp.disabled = !editable()
      inp.style.opacity = editable() ? '1' : '0.7'
      if (document.activeElement !== inp) {
        const nv = (cur.params || {})[key]
        inp.value = nv !== undefined ? nv : ''
      }
    }
    sync()
    refreshers.push(sync)
    wrap.append(caption, inp)
    return wrap
  }

  const field = (label, control) => {
    const wrap = h('div', 'margin-bottom:6px')
    wrap.append(h('label', LABEL_CSS, label), control)
    return wrap
  }

  // 普通数值参数控件；若被 wired 则回退到绑定控件
  const numberControl = (key, placeholder) => {
    const raw = (cur.params || {})[key]
    if (isWired(raw)) return wiredControl(key, raw)
    const inp = h('input', INPUT_CSS)
    inp.type = 'number'
    inp.value = raw !== undefined ? raw : ''
    inp.placeholder = placeholder || ''
    inp.addEventListener('change', () => setParam(key, inp.value))
    refreshers.push(() => {
      inp.disabled = !editable()
      inp.style.opacity = editable() ? '1' : '0.55'
      if (document.activeElement === inp) return
      const nv = (cur.params || {})[key]
      inp.value = nv !== undefined ? nv : ''
    })
    return inp
  }

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px;flex-shrink:0')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '🎯 局部重绘')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  // 重绘参数
  const denoInp = numberControl('denoise', '0.4')
  const stepsInp = numberControl('steps', '20')
  const confInp = numberControl('conf', '0.3')
  const row2 = h('div', 'display:flex;gap:6px')
  const sWrap = h('div', 'flex:1')
  const cWrap = h('div', 'flex:1')
  sWrap.append(h('label', LABEL_CSS, '采样步数 steps'), stepsInp)
  cWrap.append(h('label', LABEL_CSS, '检测阈值 conf'), confInp)
  row2.append(sWrap, cWrap)

  // detector 下拉（被 wired 时回退到绑定控件）
  let detectorCtl
  {
    const raw = (cur.params || {}).detector
    if (isWired(raw)) {
      detectorCtl = wiredControl('detector', raw)
    } else {
      const sel = h('select', INPUT_CSS)
      DETECTORS.forEach((m) => {
        const opt = h('option', '', m === 'face' ? 'face（脸部）' : 'hand（手部）')
        opt.value = m
        sel.appendChild(opt)
      })
      sel.value = raw || 'face'
      sel.addEventListener('change', () => setParam('detector', sel.value))
      refreshers.push(() => {
        sel.disabled = !editable()
        sel.style.opacity = editable() ? '1' : '0.55'
        const nv = (cur.params || {}).detector
        if (nv && DETECTORS.indexOf(nv) >= 0 && document.activeElement !== sel) sel.value = nv
      })
      detectorCtl = sel
    }
  }

  const controls = h('div', 'margin-bottom:6px;flex-shrink:0')
  controls.append(
    field('检测目标 detector', detectorCtl),
    field('重绘强度 denoise（修脸 0.35~0.5 / 修手 0.45~0.6）', denoInp),
    row2,
    h('div', 'height:6px'),
  )

  // 输出图预览
  const imgWrap = h('div', [
    'flex:1', 'min-height:0', 'display:flex', 'align-items:center', 'justify-content:center',
    'background: rgba(0,0,0,0.25)', 'border-radius: 8px', 'overflow: hidden',
    'position: relative',
  ].join(';'))
  const placeholder = h('span', 'color:rgba(255,255,255,0.25)', '等待重绘结果…')
  const img = h('img', 'max-width:100%;max-height:100%;object-fit:contain;display:none')
  img.alt = 'upscaled image'
  const zoomHint = h('span', [
    'position:absolute', 'right:6px', 'bottom:6px', 'display:none',
    'background:rgba(0,0,0,0.55)', 'border:1px solid rgba(255,255,255,0.15)',
    'border-radius:6px', 'padding:1px 6px', 'font-size:9px',
    'color:rgba(255,255,255,0.75)', 'pointer-events:none',
  ].join(';'), '🔍 点击放大')
  imgWrap.append(placeholder, img, zoomHint)

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
    const o = (cur && cur.outputs) || {}
    lightbox = buildImageLightbox(img.getAttribute('src'), 'image: ' + (o.image || '') + (o.count ? ' · 重绘 ' + o.count + ' 处' : ''), closeLightbox)
    document.addEventListener('keydown', onLightboxKey, true)
  })

  const imageLine = h('div', 'color:rgba(255,255,255,0.4);margin-top:6px;word-break:break-all;font-size:10px;flex-shrink:0')

  el.append(header, controls, imgWrap, imageLine)

  img.addEventListener('error', () => {
    img.style.display = 'none'
    img.removeAttribute('src')
    placeholder.style.display = ''
    placeholder.textContent = '已重绘（无法在线预览，对象 ID 见下方）'
    zoomHint.style.display = 'none'
    imgWrap.style.cursor = ''
    imgWrap.title = ''
  })
  img.addEventListener('load', () => {
    img.style.display = 'block'
    placeholder.style.display = 'none'
    zoomHint.style.display = ''
    imgWrap.style.cursor = 'zoom-in'
    imgWrap.title = '点击放大查看'
  })

  // wired 的 service_url 用 paramSources 的运行时值解析（与 load-image 同策略）
  function resolveServiceUrl(p) {
    const raw = (p.params || {}).service_url || ''
    if (!raw) return ''
    if (!isWired(raw)) return raw.replace(/\/+$/, '')
    const src = (p.paramSources || {}).service_url
    if (src && src.kind === 'workflow' && src.paramValue) return String(src.paramValue).replace(/\/+$/, '')
    if (src && src.kind === 'node' && src.runtimeValue) return String(src.runtimeValue).replace(/\/+$/, '')
    return ''
  }

  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const o = p.outputs || {}
    const base = resolveServiceUrl(p)
    if (o.image && base) {
      const src = base + '/images/' + o.image + '?thumb=320'
      if (img.getAttribute('src') !== src) img.src = src
    } else if (o.image) {
      // 有对象 ID 但 service_url 未解析出：只显示 ID
      img.style.display = 'none'
      placeholder.style.display = ''
      placeholder.textContent = '已重绘（service_url 未解析，无法预览）'
      zoomHint.style.display = 'none'
    } else {
      img.style.display = 'none'
      img.removeAttribute('src')
      placeholder.style.display = ''
      placeholder.textContent = p.status === 'running' ? '检测重绘中…' : '等待重绘结果…'
      zoomHint.style.display = 'none'
      imgWrap.style.cursor = ''
      imgWrap.title = ''
    }
    imageLine.textContent = o.image ? `image: ${o.image}${o.count ? ' · 重绘 ' + o.count + ' 处' : ''}` : ''
    refreshers.forEach((f) => f())
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
