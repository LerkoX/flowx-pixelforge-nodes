/**
 * save-image 画布组件：保存参数控件（filename_prefix / output_dir / index）+ 成图内嵌预览。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 *   props.params — 当前 config.params 绑定（模板值只读展示）
 *   props.paramSources — 参数绑定来源标注（可选）：workflow=流水线参数(含当前值) / node=上游节点(含显示名与运行时值) / literal=字面值
 *   props.onParamsChange(params) — 全量写回该节点参数（回放态缺省 = 控件只读）
 */

const STATUS_COLORS = {
  idle: '#94a3b8',
  running: '#22d3ee',
  success: '#34d399',
  failed: '#fb7185',
  skipped: '#64748b',
}

const INPUT_CSS = 'width:100%;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.14);border-radius:6px;padding:4px 7px;font-size:11px;color:rgba(255,255,255,0.9);outline:none;box-sizing:border-box;font-family:inherit'
const LABEL_CSS = 'display:block;font-size:9px;color:rgba(255,255,255,0.35);margin-bottom:2px'
const WIRED_CSS = 'font-size:10px;color:rgba(165,180,252,0.75);font-family:ui-monospace,Menlo,monospace;background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.22);border-radius:6px;padding:3px 7px;word-break:break-all'

function h(tag, style, text) {
  const n = document.createElement(tag)
  if (style) n.style.cssText = style
  if (text !== undefined) n.textContent = text
  return n
}

const isWired = (v) => typeof v === 'string' && v.indexOf('{{') >= 0

// Studio 本地媒体文件端点（同源，浏览器自动携带认证 cookie）
const mediaUrl = (p) => '/api/v1/media/file?path=' + encodeURIComponent(p)

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
  const refreshers = []
  const editable = () => typeof cur.onParamsChange === 'function'
  const setParam = (key, value) => {
    if (!editable()) return
    const params = { ...(cur.params || {}), [key]: String(value) }
    cur = { ...cur, params }
    cur.onParamsChange(params)
  }

  // wired 参数控件（{{ ... }} 绑定）：来源标签 + 可编辑文本框。
  // 来源标签优先取 props.paramSources（Studio 解析 YAML 下发）：
  //   workflow → "⚡ 流水线参数 · name = 当前值"（随参数面板实时更新）
  //   node     → "🔗 节点显示名 · field = 运行时值"（执行中/回放有上游输出时）
  //   literal  → "✏️ 自定义值"（解除绑定后）；无 paramSources 时回退展示原始绑定串。
  // 输入普通值 = 解除绑定；输入 {{ Param.xxx }} / {{ 节点.字段 }} = 重新绑定。
  const WIRED_INPUT_CSS = 'width:100%;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:6px;padding:4px 7px;font-size:10px;color:rgba(165,180,252,0.95);outline:none;box-sizing:border-box;font-family:ui-monospace,Menlo,monospace'
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

  const textControl = (key, placeholder, type) => {
    const raw = (cur.params || {})[key]
    if (isWired(raw)) return wiredControl(key, raw)
    const inp = h('input', INPUT_CSS)
    inp.type = type || 'text'
    if (type === 'number') inp.min = '0'
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
  const title = h('strong', 'font-size:12px', '💾 Save Image')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  // 保存参数
  const controls = h('div', 'margin-bottom:6px;flex-shrink:0')
  controls.append(
    field('文件名前缀 filename_prefix', textControl('filename_prefix', 'flowx')),
    field('输出目录 output_dir', textControl('output_dir', '~/flowx-output')),
    field('起始序号 index', textControl('index', '0', 'number')),
  )

  const imgWrap = h('div', [
    'flex:1', 'min-height:0', 'display:flex', 'align-items:center', 'justify-content:center',
    'background: rgba(0,0,0,0.25)', 'border-radius: 8px', 'overflow: hidden',
    'position: relative',
  ].join(';'))
  const placeholder = h('span', 'color:rgba(255,255,255,0.25)', '等待成图…')
  const img = h('img', 'max-width:100%;max-height:100%;object-fit:contain;display:none')
  img.alt = 'generated image'
  const zoomHint = h('span', [
    'position:absolute', 'right:6px', 'bottom:6px', 'display:none',
    'background:rgba(0,0,0,0.55)', 'border:1px solid rgba(255,255,255,0.15)',
    'border-radius:6px', 'padding:1px 6px', 'font-size:9px',
    'color:rgba(255,255,255,0.75)', 'pointer-events:none',
  ].join(';'), '🔍 点击放大')
  imgWrap.append(placeholder, img, zoomHint)

  // 点击缩略图弹出可缩放 lightbox（滚轮/双击/触屏捏合，见 buildImageLightbox）。
  // 图片经 /api/v1/media/file 直读本地文件（原图无损），不走 base64。
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
  const openLightbox = () => {
    const o = (cur && cur.outputs) || {}
    if (!o.file_path) return
    closeLightbox()
    lightbox = buildImageLightbox(mediaUrl(o.file_path), o.file_path, closeLightbox)
    document.addEventListener('keydown', onLightboxKey, true)
  }
  imgWrap.addEventListener('click', () => {
    if (img.style.display !== 'none') openLightbox()
  })

  const pathLine = h('div', 'color:rgba(255,255,255,0.4);margin-top:6px;word-break:break-all;font-size:10px;flex-shrink:0')

  el.append(header, controls, imgWrap, pathLine)

  // 直读本地文件失败（文件不在 server 媒体白名单/已被清理）→ 占位提示
  img.addEventListener('error', () => {
    img.style.display = 'none'
    img.removeAttribute('src')
    placeholder.style.display = ''
    placeholder.textContent = '已保存（无法在线预览，见下方路径）'
    zoomHint.style.display = 'none'
    imgWrap.style.cursor = ''
    imgWrap.title = ''
  })

  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const o = p.outputs || {}
    if (o.file_path) {
      // 直读本地文件：原图无损，不占输出通道体积
      const src = mediaUrl(o.file_path)
      if (img.getAttribute('src') !== src) img.src = src
      img.style.display = 'block'
      placeholder.style.display = 'none'
      zoomHint.style.display = ''
      imgWrap.style.cursor = 'zoom-in'
      imgWrap.title = '点击放大查看'
    } else {
      img.style.display = 'none'
      placeholder.style.display = ''
      placeholder.textContent = p.status === 'running' ? '生成中…' : '等待成图…'
      zoomHint.style.display = 'none'
      imgWrap.style.cursor = ''
      imgWrap.title = ''
    }
    pathLine.textContent = o.file_path || ''
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
