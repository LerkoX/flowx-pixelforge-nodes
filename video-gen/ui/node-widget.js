/**
 * video-gen 画布组件：视频生成参数（prompt/尺寸/帧数/fps/steps/cfg/seed）+ 采样进度。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 *   props.preview — 服务端经 Studio 回调推的进度卡片帧（{ image, mime, progress }），
 *   瞬态，节点完成即清除，需判空
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
const WIRED_INPUT_CSS = 'width:100%;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:6px;padding:4px 7px;font-size:10px;color:rgba(165,180,252,0.95);outline:none;box-sizing:border-box;font-family:ui-monospace,Menlo,monospace'

function h(tag, style, text) {
  const n = document.createElement(tag)
  if (style) n.style.cssText = style
  if (text !== undefined) n.textContent = text
  return n
}

const isWired = (v) => typeof v === 'string' && v.indexOf('{{') >= 0

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

  let cur = props
  const refreshers = []
  const editable = () => typeof cur.onParamsChange === 'function'
  const setParam = (key, value) => {
    if (!editable()) return
    const params = { ...(cur.params || {}), [key]: String(value) }
    cur = { ...cur, params }
    cur.onParamsChange(params)
  }

  const wiredControl = (key, raw) => {
    const wrap = h('div', '')
    const caption = h('div', 'font-size:9px;margin-bottom:2px;word-break:break-all;line-height:1.4;color:rgba(165,180,252,0.75)')
    const inp = h('input', WIRED_INPUT_CSS)
    inp.type = 'text'
    inp.value = raw !== undefined ? raw : ''
    inp.spellcheck = false
    inp.title = '参数绑定：输入普通值解除绑定；输入 {{ Param.xxx }} 或 {{ 节点.字段 }} 重新绑定'
    inp.addEventListener('change', () => setParam(key, inp.value.trim()))
    const sync = () => {
      caption.textContent = '⟵ ' + ((cur.params || {})[key] ?? '')
      inp.disabled = !editable()
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

  const sliderControl = (key, min, max, step, def, fmt, valWidth) => {
    const raw = (cur.params || {})[key]
    if (isWired(raw)) return wiredControl(key, raw)
    const num = (x) => { const n = parseFloat(x); return isNaN(n) ? def : n }
    const wrap = h('div', 'display:flex;align-items:center;gap:6px')
    const sl = h('input', 'flex:1;accent-color:#818cf8;margin:0;min-width:0')
    sl.type = 'range'; sl.min = min; sl.max = max; sl.step = step
    const val = h('span', `font-size:10px;color:rgba(255,255,255,0.6);min-width:${valWidth || 30}px;text-align:right;font-family:ui-monospace,Menlo,monospace`)
    sl.value = String(num(raw))
    val.textContent = fmt(num(raw))
    sl.addEventListener('input', () => { val.textContent = fmt(parseFloat(sl.value)) })
    sl.addEventListener('change', () => setParam(key, sl.value))
    refreshers.push(() => {
      sl.disabled = !editable()
      sl.style.opacity = editable() ? '1' : '0.55'
      const nv = num((cur.params || {})[key])
      sl.value = String(nv)
      val.textContent = fmt(nv)
    })
    wrap.append(sl, val)
    return wrap
  }

  // prompt：多行文本（wired 时退化为绑定框）
  const promptControl = (() => {
    const raw = (cur.params || {}).prompt
    if (isWired(raw)) return wiredControl('prompt', raw)
    const ta = h('textarea', INPUT_CSS + ';resize:vertical;min-height:44px;line-height:1.4')
    ta.value = raw !== undefined ? raw : ''
    ta.spellcheck = false
    ta.addEventListener('change', () => setParam('prompt', ta.value))
    refreshers.push(() => {
      ta.disabled = !editable()
      ta.style.opacity = editable() ? '1' : '0.7'
      if (document.activeElement !== ta) {
        const nv = (cur.params || {}).prompt
        ta.value = nv !== undefined ? nv : ''
      }
    })
    return ta
  })()

  // seed：数字输入 + 随机按钮
  const seedControl = (() => {
    const raw = (cur.params || {}).seed
    if (isWired(raw)) return wiredControl('seed', raw)
    const wrap = h('div', 'display:flex;gap:6px')
    const inp = h('input', INPUT_CSS)
    inp.type = 'number'; inp.step = '1'
    inp.value = raw !== undefined ? raw : '-1'
    inp.title = '-1 表示每次随机'
    const dice = h('button', 'flex-shrink:0;width:28px;border-radius:6px;border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.06);cursor:pointer;font-size:12px;line-height:1;color:rgba(255,255,255,0.8)', '🎲')
    inp.addEventListener('change', () => setParam('seed', inp.value))
    dice.addEventListener('click', () => {
      const v = String(Math.floor(Math.random() * 2 ** 32))
      inp.value = v
      setParam('seed', v)
    })
    refreshers.push(() => {
      inp.disabled = !editable(); dice.disabled = !editable()
      if (document.activeElement === inp) return
      const nv = (cur.params || {}).seed
      inp.value = nv !== undefined ? nv : '-1'
    })
    wrap.append(inp, dice)
    return wrap
  })()

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '🎬 视频生成')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  // 进度面板：running 时显示服务端推来的进度卡片帧 + 进度条
  const previewBox = h('div', 'position:relative;width:100%;aspect-ratio:16/9;border:1px dashed rgba(255,255,255,0.15);border-radius:8px;background:rgba(0,0,0,0.25);overflow:hidden;margin-bottom:6px;display:flex;align-items:center;justify-content:center;box-sizing:border-box')
  const previewPlaceholder = h('span', 'font-size:10px;color:rgba(255,255,255,0.25);padding:8px;text-align:center', '等待执行…')
  const previewImg = h('img', 'display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain')
  previewImg.draggable = false
  const previewBar = h('div', 'display:none;position:absolute;left:0;right:0;bottom:0;height:3px;background:rgba(0,0,0,0.5)')
  const previewBarFill = h('div', 'height:100%;background:#22d3ee;transition:width .3s ease-out;width:0%')
  previewBar.append(previewBarFill)
  previewBox.append(previewPlaceholder, previewImg, previewBar)

  const sizeRow = h('div', 'display:grid;grid-template-columns:1fr 1fr;gap:6px')
  sizeRow.append(
    field('宽 width', sliderControl('width', 256, 1280, 32, 832, (v) => String(v), 34)),
    field('高 height', sliderControl('height', 256, 1280, 32, 480, (v) => String(v), 34)),
  )
  const lenRow = h('div', 'display:grid;grid-template-columns:1fr 1fr;gap:6px')
  lenRow.append(
    field('帧数 frames', sliderControl('num_frames', 17, 241, 8, 121, (v) => String(v), 34)),
    field('帧率 fps', sliderControl('fps', 8, 30, 1, 24, (v) => String(v))),
  )

  const controls = h('div', 'margin-bottom:6px')
  controls.append(
    field('提示词 prompt', promptControl),
    sizeRow,
    lenRow,
    field('步数 steps', sliderControl('steps', 10, 80, 5, 50, (v) => String(v))),
    field('引导强度 cfg', sliderControl('cfg', 1, 15, 0.5, 5, (v) => v.toFixed(1))),
    field('种子 seed（-1 随机）', seedControl),
  )

  const seedLine = h('div', 'color:#fbbf24;font-weight:600')
  const videoLine = h('div', 'color:rgba(255,255,255,0.45);word-break:break-all')

  el.append(header, previewBox, controls, seedLine, videoLine)

  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const o = p.outputs || {}
    seedLine.textContent = o.seed ? `seed: ${o.seed}` : ''
    videoLine.textContent = o.video
      ? `video: ${o.video}`
      : (p.status === 'running' ? '视频采样中（分钟级）…' : '等待执行…')
    const pv = p.preview
    if (pv && pv.image) {
      previewBox.style.display = 'flex'
      previewImg.src = 'data:' + (pv.mime || 'image/jpeg') + ';base64,' + pv.image
      previewImg.style.display = 'block'
      previewPlaceholder.style.display = 'none'
      previewBar.style.display = 'block'
      previewBarFill.style.width = Math.round((pv.progress || 0) * 100) + '%'
    } else if (p.status === 'running' || p.status === 'idle') {
      previewBox.style.display = 'flex'
      previewImg.style.display = 'none'
      previewBar.style.display = 'none'
      previewPlaceholder.style.display = ''
      previewPlaceholder.textContent = p.status === 'running'
        ? '采样中…（preview_every>0 时显示进度卡片）'
        : '等待执行…'
    } else {
      previewBox.style.display = 'none'
    }
    refreshers.forEach((f) => f())
  }

  render(props)

  return {
    update(next) { render(next) },
    unmount() { el.textContent = '' },
  }
}
