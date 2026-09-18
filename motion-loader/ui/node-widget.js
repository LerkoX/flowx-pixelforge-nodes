/**
 * motion-loader 画布组件：Motion 组合卡片（底模 + 运动模块 → model/clip/vae 引用）+ 加载进度。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 *   props.params — 当前 config.params 绑定（模板值只读展示）
 *   props.paramSources — 参数绑定来源标注（可选）：workflow=流水线参数(含当前值) / node=上游节点(含显示名与运行时值) / literal=字面值
 *   props.onParamsChange(params) — 全量写回该节点参数（回放态缺省 = 控件只读）
 * 说明：motion.load 是服务端原子任务（无逐步进度），running 时用不定态进度条表示组合中；
 *   完成后展示 model_ref（高亮）与 clip/vae 引用。
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

  // wired 参数控件（{{ ... }} 绑定）：来源标签 + 可编辑文本框（与 checkpoint-loader 同款交互）
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

  // 普通文本参数控件：wired 时退化为绑定框
  const textControl = (key, placeholder) => {
    const raw = (cur.params || {})[key]
    if (isWired(raw)) return wiredControl(key, raw)
    const inp = h('input', INPUT_CSS)
    inp.type = 'text'
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

  const field = (label, control) => {
    const wrap = h('div', 'margin-bottom:6px')
    wrap.append(h('label', LABEL_CSS, label), control)
    return wrap
  }

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '🎞️ Motion 组合')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  // 加载进度：motion.load 是服务端原子任务（无逐步进度），running 时显示不定态进度条
  const progressBox = h('div', 'display:none;margin-bottom:6px')
  const progressText = h('div', 'font-size:10px;color:rgba(34,211,238,0.85);margin-bottom:3px')
  const progressTrack = h('div', 'position:relative;height:4px;border-radius:2px;background:rgba(255,255,255,0.08);overflow:hidden')
  const progressFill = h('div', 'position:absolute;top:0;bottom:0;left:0;width:40%;border-radius:2px;background:#22d3ee')
  progressTrack.append(progressFill)
  progressBox.append(progressText, progressTrack)
  // 不定态滑动动画（注入一次 keyframes）
  const styleTag = document.createElement('style')
  styleTag.textContent = '@keyframes flowx-motion-slide{0%{left:-40%}100%{left:100%}}'
  el.appendChild(styleTag)
  progressFill.style.animation = 'flowx-motion-slide 1.2s ease-in-out infinite'

  const modelLine = h('div', 'color:#a5b4fc;font-weight:600;word-break:break-all')
  const refsBox = h('div', 'color:rgba(255,255,255,0.4);margin-top:2px')

  el.append(
    header,
    field('SD1.x 底模 ckpt_name', textControl('ckpt_name', '如 v1-5-pruned-emaonly')),
    field('运动模块 motion_name', textControl('motion_name', '如 mm_sd_v15_v2')),
    progressBox,
    modelLine,
    refsBox,
  )

  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const o = p.outputs || {}
    if (p.status === 'running') {
      progressBox.style.display = 'block'
      progressText.textContent = '组合中：底模 + MotionAdapter → AnimateDiffPipeline（分钟级）…'
      modelLine.textContent = ''
      refsBox.textContent = ''
    } else {
      progressBox.style.display = 'none'
      modelLine.textContent = o.model_ref ? `model: ${o.model_ref}` : '未加载'
      refsBox.textContent = o.model_ref
        ? `clip=${o.clip_ref || '-'}  vae=${o.vae_ref || '-'}`
        : ''
    }
    refreshers.forEach((f) => f())
  }

  render(props)

  return {
    update(next) { render(next) },
    unmount() { el.textContent = '' },
  }
}
