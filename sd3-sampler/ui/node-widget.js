/**
 * sd3-sampler 画布组件：flow matching 采样参数面板 + 进度卡片预览。
 * 契约：默认导出 mount(el, props) => { update(props), unmount() }（apiVersion 1）。
 * props.preview：{ url, progress } 进度卡片帧（16ch latent 无法投影，服务端推卡片）；
 * props.paramSources：参数绑定来源标注；props.onParamsChange：参数写回（回放态缺省）。
 */

const STATUS_COLORS = {
  idle: '#94a3b8', running: '#22d3ee', success: '#34d399',
  failed: '#fb7185', skipped: '#64748b',
}

export default function mount(el, props) {
  let cur = props || {}
  el.style.cssText = 'font-family:system-ui,sans-serif;background:rgba(255,255,255,0.04);'
    + 'border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:10px;font-size:11px;'
    + 'line-height:1.7;color:rgba(255,255,255,0.85);overflow:auto;box-sizing:border-box;height:100%'

  const h = (tag, style, text) => {
    const n = document.createElement(tag)
    if (style) n.style.cssText = style
    if (text !== undefined) n.textContent = text
    return n
  }

  // header
  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '🌊 SD3 采样器')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  // 预览面板（进度卡片帧）
  const previewBox = h('div', 'position:relative;width:100%;aspect-ratio:1/1;border:1px dashed '
    + 'rgba(255,255,255,0.15);border-radius:8px;background:rgba(0,0,0,0.25);overflow:hidden;'
    + 'margin-bottom:6px;display:flex;align-items:center;justify-content:center;box-sizing:border-box')
  const previewPlaceholder = h('span', 'font-size:10px;color:rgba(255,255,255,0.25);padding:8px;text-align:center',
    '等待执行…\n（预览为进度卡片：16ch latent 无法投影成图）')
  const previewImg = h('img', 'display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain')
  previewImg.draggable = false
  const previewBar = h('div', 'display:none;position:absolute;left:0;right:0;bottom:0;height:3px;background:rgba(0,0,0,0.5)')
  const previewBarFill = h('div', 'height:100%;background:#22d3ee;transition:width .3s ease-out;width:0%')
  previewBar.append(previewBarFill)
  previewBox.append(previewPlaceholder, previewImg, previewBar)

  // 参数区
  const paramsBox = h('div', 'margin-bottom:6px')
  const PARAM_LABELS = { seed: '种子', steps: '步数', cfg: 'CFG', denoise: '去噪' }
  const PARAM_KEYS = ['steps', 'cfg', 'denoise', 'seed']

  // 输出区
  const outputsBox = h('div', 'border-top:1px solid rgba(255,255,255,0.08);padding-top:6px')

  el.append(header, previewBox, paramsBox, outputsBox)

  const editable = () => typeof cur.onParamsChange === 'function'
  const sourceCaptionOf = (key) => {
    const src = (cur.paramSources || {})[key]
    if (!src) return null
    if (src.kind === 'workflow') return `⚡ 流水线参数 · ${src.paramName ?? ''}`
    if (src.kind === 'node') return `🔗 ${src.nodeName ?? '上游节点'} · ${src.output ?? ''}`
    return '✏️ 自定义值'
  }

  function renderParams() {
    paramsBox.textContent = ''
    const params = cur.params || {}
    for (const key of PARAM_KEYS) {
      const row = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:2px')
      const label = h('span', 'color:rgba(255,255,255,0.5);width:34px', PARAM_LABELS[key])
      const input = h('input', 'flex:1;background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.12);'
        + 'border-radius:4px;color:inherit;font-size:11px;padding:2px 6px;min-width:0')
      input.value = params[key] ?? ''
      input.disabled = !editable()
      input.onchange = () => {
        if (!editable()) return
        const next = Object.assign({}, params, { [key]: input.value })
        cur.onParamsChange(next)
      }
      row.append(label, input)
      const cap = sourceCaptionOf(key)
      if (cap) row.append(h('span', 'color:rgba(255,255,255,0.3);font-size:9px;white-space:nowrap', cap))
      paramsBox.append(row)
    }
  }

  function renderOutputs() {
    outputsBox.textContent = ''
    const outs = cur.outputs || {}
    const seed = outs.seed ?? outs['seed']
    if (seed !== undefined) {
      outputsBox.append(h('div', 'color:rgba(255,255,255,0.55)',
        `seed: ${seed}`))
    }
    if (outs.latent) {
      outputsBox.append(h('div', 'color:rgba(255,255,255,0.3);font-size:9px;word-break:break-all',
        `latent: ${outs.latent}`))
    }
  }

  function render() {
    const st = cur.status || 'idle'
    dot.style.background = STATUS_COLORS[st] || STATUS_COLORS.idle
    statusLabel.textContent = st
    const pv = cur.preview
    if (pv && pv.url) {
      previewImg.src = pv.url
      previewImg.style.display = 'block'
      previewPlaceholder.style.display = 'none'
      previewBar.style.display = 'block'
      previewBarFill.style.width = `${Math.round((pv.progress || 0) * 100)}%`
    } else {
      previewImg.style.display = 'none'
      previewPlaceholder.style.display = ''
      previewBar.style.display = 'none'
    }
    renderParams()
    renderOutputs()
  }

  render()
  return {
    update(next) { cur = next || {}; render() },
    unmount() { el.textContent = '' },
  }
}
