/**
 * sd3-clip-encode 画布组件：提示词文本 + 编码统计（embed 形状/范数）展示。
 * 契约：默认导出 mount(el, props) => { update(props), unmount() }（apiVersion 1）。
 */
const STATUS_COLORS = { idle: '#94a3b8', running: '#22d3ee', success: '#34d399', failed: '#fb7185', skipped: '#64748b' }

export default function mount(el, props) {
  let cur = props || {}
  el.style.cssText = 'font-family:system-ui,sans-serif;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:10px;font-size:11px;line-height:1.7;color:rgba(255,255,255,0.85);overflow:auto;box-sizing:border-box;height:100%'
  const h = (tag, style, text) => { const n = document.createElement(tag); if (style) n.style.cssText = style; if (text !== undefined) n.textContent = text; return n }

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '📝 SD3 CLIP 编码')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  const promptBox = h('div', 'background:rgba(0,0,0,0.25);border-radius:8px;padding:8px;margin-bottom:6px;word-break:break-word;white-space:pre-wrap;max-height:90px;overflow:auto')
  const promptCap = h('div', 'color:rgba(255,255,255,0.3);font-size:9px;margin-bottom:6px')
  const infoBox = h('div', 'border-top:1px solid rgba(255,255,255,0.08);padding-top:6px;color:rgba(34,211,238,0.8);font-size:10px;word-break:break-all')
  el.append(header, promptBox, promptCap, infoBox)

  function render() {
    const st = cur.status || 'idle'
    dot.style.background = STATUS_COLORS[st] || STATUS_COLORS.idle
    statusLabel.textContent = st
    const text = (cur.params || {}).text ?? ''
    promptBox.textContent = text || '（未设置提示词）'
    const src = (cur.paramSources || {}).text
    promptCap.textContent = src
      ? (src.kind === 'workflow' ? `⚡ 流水线参数 · ${src.paramName ?? ''}`
        : src.kind === 'node' ? `🔗 ${src.nodeName ?? '上游'} · ${src.output ?? ''}` : '✏️ 自定义值')
      : ''
    const info = (cur.outputs || {}).info
    infoBox.textContent = info ? `📊 ${info}` : ''
  }
  render()
  return { update(next) { cur = next || {}; render() }, unmount() { el.textContent = '' } }
}
