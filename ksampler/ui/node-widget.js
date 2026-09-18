/**
 * ksampler 画布组件：采样参数控件（seed/steps/cfg/sampler/denoise）+ 采样状态。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 *   props.params — 当前 config.params 绑定（模板值只读展示）
 *   props.paramSources — 参数绑定来源标注（可选）：workflow=流水线参数(含当前值) / node=上游节点(含显示名与运行时值) / literal=字面值
 *   props.onParamsChange(params) — 全量写回该节点参数（回放态缺省 = 控件只读）
 *   props.preview — 实时预览帧（可选）：{ url, progress }，采样中由推理服务
 *   把帧留在 GET /preview/{job_id}，Studio 中转后随 SSE 进度事件刷新 url；
 *   瞬态，节点完成即清除，需判空；<img src=url> 直出（媒体不经 base64）
 */

const STATUS_COLORS = {
  idle: '#94a3b8',
  running: '#22d3ee',
  success: '#34d399',
  failed: '#fb7185',
  skipped: '#64748b',
}

const SAMPLERS = ['euler', 'euler_a', 'ddim', 'lms', 'dpmpp_2m', 'dpmpp_2m_karras', 'dpmpp_2m_sde', 'uni_pc']

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

  // seed：数字输入 + 随机按钮（-1 = 每次随机）
  const seedControl = (() => {
    const raw = (cur.params || {}).seed
    if (isWired(raw)) return wiredControl('seed', raw)
    const wrap = h('div', 'display:flex;gap:6px')
    const inp = h('input', INPUT_CSS)
    inp.type = 'number'
    inp.step = '1'
    inp.value = raw !== undefined ? raw : '-1'
    inp.title = '-1 表示每次随机'
    const dice = h('button', 'flex-shrink:0;width:28px;border-radius:6px;border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.06);cursor:pointer;font-size:12px;line-height:1;color:rgba(255,255,255,0.8)', '🎲')
    dice.title = '随机种子'
    inp.addEventListener('change', () => setParam('seed', inp.value))
    dice.addEventListener('click', () => {
      const v = String(Math.floor(Math.random() * 2 ** 32))
      inp.value = v
      setParam('seed', v)
    })
    refreshers.push(() => {
      inp.disabled = !editable()
      dice.disabled = !editable()
      inp.style.opacity = editable() ? '1' : '0.55'
      dice.style.opacity = editable() ? '1' : '0.4'
      if (document.activeElement === inp) return
      const nv = (cur.params || {}).seed
      inp.value = nv !== undefined ? nv : '-1'
    })
    wrap.append(inp, dice)
    return wrap
  })()

  // sampler 下拉
  const samplerControl = (() => {
    const raw = (cur.params || {}).sampler_name
    if (isWired(raw)) return wiredControl('sampler_name', raw)
    const sel = h('select', INPUT_CSS + ';appearance:auto')
    for (const name of SAMPLERS) {
      const opt = h('option', '', name)
      opt.value = name
      sel.appendChild(opt)
    }
    sel.value = raw !== undefined ? raw : 'euler'
    sel.addEventListener('change', () => setParam('sampler_name', sel.value))
    refreshers.push(() => {
      sel.disabled = !editable()
      sel.style.opacity = editable() ? '1' : '0.55'
      const nv = (cur.params || {}).sampler_name
      sel.value = nv !== undefined ? nv : 'euler'
    })
    return sel
  })()

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '🎲 KSampler')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  // 实时预览面板（props.preview：采样中由推理服务经 Studio 回调逐步推送，
  // 瞬态——节点完成即清除；未执行/回放态显示占位框）
  const previewBox = h('div', 'position:relative;width:100%;aspect-ratio:1/1;border:1px dashed rgba(255,255,255,0.15);border-radius:8px;background:rgba(0,0,0,0.25);overflow:hidden;margin-bottom:6px;display:flex;align-items:center;justify-content:center;box-sizing:border-box')
  const previewPlaceholder = h('span', 'font-size:10px;color:rgba(255,255,255,0.25);padding:8px;text-align:center', '等待执行…')
  const previewImg = h('img', 'display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain')
  previewImg.draggable = false
  const previewBar = h('div', 'display:none;position:absolute;left:0;right:0;bottom:0;height:3px;background:rgba(0,0,0,0.5)')
  const previewBarFill = h('div', 'height:100%;background:#22d3ee;transition:width .3s ease-out;width:0%')
  previewBar.append(previewBarFill)
  previewBox.append(previewPlaceholder, previewImg, previewBar)

  const controls = h('div', 'margin-bottom:6px')
  controls.append(
    field('种子 seed（-1 随机）', seedControl),
    field('步数 steps', sliderControl('steps', 1, 50, 1, 20, (v) => String(v))),
    field('引导强度 cfg', sliderControl('cfg', 1, 20, 0.5, 7, (v) => v.toFixed(1))),
    field('采样器 sampler', samplerControl),
    field('去噪 denoise', sliderControl('denoise', 0, 1, 0.05, 1, (v) => v.toFixed(2), 34)),
  )

  const seedLine = h('div', 'color:#fbbf24;font-weight:600')
  const latentLine = h('div', 'color:rgba(255,255,255,0.45);word-break:break-all')

  el.append(header, previewBox, controls, seedLine, latentLine)

  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const o = p.outputs || {}
    seedLine.textContent = o.seed ? `seed: ${o.seed}` : ''
    latentLine.textContent = o.latent
      ? `latent: ${o.latent}`
      : (p.status === 'running' ? '采样中…' : '等待执行…')
    // 预览面板：有帧显示图像 + 进度条；running 无帧提示等待（推理服务需 ≥1.2）；
    // 终态无帧时整块隐藏（输出区已展示 seed/latent）
    const pv = p.preview
    if (pv && pv.url) {
      previewBox.style.display = 'flex'
      previewImg.src = pv.url
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
        ? '采样中，等待预览帧…（需推理服务 ≥1.2）'
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
