/**
 * sd3-empty-latent 画布组件：宽高/批次参数 + latent 形状信息。
 * 契约：默认导出 mount(el, props) => { update(props), unmount() }（apiVersion 1）。
 */
const STATUS_COLORS = { idle: '#94a3b8', running: '#22d3ee', success: '#34d399', failed: '#fb7185', skipped: '#64748b' }

export default function mount(el, props) {
  let cur = props || {}
  el.style.cssText = 'font-family:system-ui,sans-serif;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:10px;font-size:11px;line-height:1.7;color:rgba(255,255,255,0.85);overflow:auto;box-sizing:border-box;height:100%'
  const h = (tag, style, text) => { const n = document.createElement(tag); if (style) n.style.cssText = style; if (text !== undefined) n.textContent = text; return n }

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '🧱 SD3 空 Latent')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  const paramsBox = h('div', 'margin-bottom:6px')
  const infoBox = h('div', 'border-top:1px solid rgba(255,255,255,0.08);padding-top:6px;color:rgba(34,211,238,0.8);font-size:10px;word-break:break-all')
  el.append(header, paramsBox, infoBox)

  const LABELS = { width: '宽', height: '高', batch_size: '批次' }
  const editable = () => typeof cur.onParamsChange === 'function'

  function render() {
    const st = cur.status || 'idle'
    dot.style.background = STATUS_COLORS[st] || STATUS_COLORS.idle
    statusLabel.textContent = st
    paramsBox.textContent = ''
    const params = cur.params || {}
    for (const key of ['width', 'height', 'batch_size']) {
      const row = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:2px')
      row.append(h('span', 'color:rgba(255,255,255,0.5);width:34px', LABELS[key]))
      const input = h('input', 'flex:1;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.12);border-radius:4px;color:inherit;font-size:11px;padding:2px 6px;min-width:0')
      input.value = params[key] ?? ''
      input.disabled = !editable()
      input.onchange = () => {
        if (!editable()) return
        cur.onParamsChange(Object.assign({}, params, { [key]: input.value }))
      }
      row.append(input)
      const src = (cur.paramSources || {})[key]
      if (src) row.append(h('span', 'color:rgba(255,255,255,0.3);font-size:9px;white-space:nowrap',
        src.kind === 'workflow' ? `⚡ ${src.paramName ?? ''}` : src.kind === 'node' ? `🔗 ${src.nodeName ?? ''}` : '✏️'))
      paramsBox.append(row)
    }
    const info = (cur.outputs || {}).info
    infoBox.textContent = info ? `📊 ${info}` : ''
  }
  render()
  return { update(next) { cur = next || {}; render() }, unmount() { el.textContent = '' } }
}
