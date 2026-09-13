/**
 * inference-op 画布组件：通用算子参数控件（op_name / inputs_json / emit_keys）+ 输出端口。
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

  const textControl = (key, placeholder, mono) => {
    const raw = (cur.params || {})[key]
    if (isWired(raw)) return wiredControl(key, raw)
    const inp = h('input', INPUT_CSS + (mono ? ';font-family:ui-monospace,Menlo,monospace' : ''))
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

  const textareaControl = (key, placeholder) => {
    const raw = (cur.params || {})[key]
    if (isWired(raw)) return wiredControl(key, raw)
    const ta = h('textarea', INPUT_CSS + ';font-family:ui-monospace,Menlo,monospace;resize:vertical;min-height:44px')
    ta.rows = 2
    ta.value = raw !== undefined ? raw : ''
    ta.placeholder = placeholder || ''
    ta.addEventListener('change', () => setParam(key, ta.value))
    refreshers.push(() => {
      ta.disabled = !editable()
      ta.style.opacity = editable() ? '1' : '0.55'
      if (document.activeElement === ta) return
      const nv = (cur.params || {})[key]
      ta.value = nv !== undefined ? nv : ''
    })
    return ta
  }

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '🧩 通用算子')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  const controls = h('div', 'margin-bottom:6px')
  controls.append(
    field('算子 op_name', textControl('op_name', 'upscale / latent.empty / …', true)),
    field('入参 inputs_json', textareaControl('inputs_json', '{"key": "value"}')),
    field('输出键 emit_keys（逗号分隔）', textControl('emit_keys', 'latent,image', true)),
  )

  const outBox = h('div', 'color:rgba(255,255,255,0.45);word-break:break-all')

  el.append(header, controls, outBox)

  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const o = p.outputs || {}
    outBox.textContent = ''
    const entries = Object.entries(o)
    if (entries.length === 0) {
      outBox.textContent = '等待执行…'
    } else {
      for (const [k, v] of entries) {
        const row = h('div')
        row.textContent = `${k}: ${String(v).slice(0, 80)}`
        outBox.append(row)
      }
    }
    refreshers.forEach((f) => f())
  }

  render(props)

  return {
    update(next) { render(next) },
    unmount() { el.textContent = '' },
  }
}
