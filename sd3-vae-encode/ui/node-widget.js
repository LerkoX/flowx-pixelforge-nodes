/**
 * sd3-vae-encode 画布组件：输入图像预览（经 service_url 拉推理服务缩略图）+ 输出 latent。
 * 契约：默认导出 mount(el, props) => { update(props), unmount() }（apiVersion 1）。
 * image 参数来自上游 load-image 绑定，经 paramSources 运行时值解析。
 */
const STATUS_COLORS = { idle: '#94a3b8', running: '#22d3ee', success: '#34d399', failed: '#fb7185', skipped: '#64748b' }

export default function mount(el, props) {
  let cur = props || {}
  let blobUrl = null
  el.style.cssText = 'font-family:system-ui,sans-serif;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:10px;font-size:11px;line-height:1.7;color:rgba(255,255,255,0.85);overflow:auto;box-sizing:border-box;height:100%'
  const h = (tag, style, text) => { const n = document.createElement(tag); if (style) n.style.cssText = style; if (text !== undefined) n.textContent = text; return n }

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '📥 SD3 VAE 编码')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  const imgBox = h('div', 'position:relative;width:100%;aspect-ratio:1/1;border:1px dashed rgba(255,255,255,0.15);border-radius:8px;background:rgba(0,0,0,0.25);overflow:hidden;display:flex;align-items:center;justify-content:center;box-sizing:border-box;margin-bottom:6px')
  const placeholder = h('span', 'font-size:10px;color:rgba(255,255,255,0.25);padding:8px;text-align:center', '等待输入图像…')
  const img = h('img', 'display:none;position:absolute;inset:0;width:100%;height:100%;object-fit:contain')
  img.draggable = false
  imgBox.append(placeholder, img)
  const idLine = h('div', 'color:rgba(255,255,255,0.3);font-size:9px;word-break:break-all')
  el.append(header, imgBox, idLine)

  const runtimeValueOf = (key) => {
    const src = (cur.paramSources || {})[key]
    if (src) return src.runtimeValue ?? src.paramValue ?? null
    const raw = (cur.params || {})[key]
    return typeof raw === 'string' && !raw.includes('{{') ? raw : null
  }

  async function loadThumb(imageId) {
    const base = runtimeValueOf('service_url')
    if (!base || !imageId || imageId.startsWith('mock-')) return
    const tok = runtimeValueOf('service_token') || ''
    try {
      const resp = await fetch(`${base.replace(/\/$/, '')}/images/${imageId}?thumb=512`,
        { headers: tok ? { Authorization: `Bearer ${tok}` } : {} })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const blob = await resp.blob()
      if (blobUrl) URL.revokeObjectURL(blobUrl)
      blobUrl = URL.createObjectURL(blob)
      img.src = blobUrl
      img.style.display = 'block'
      placeholder.style.display = 'none'
    } catch (e) {
      placeholder.textContent = `缩略图加载失败：${e.message}`
    }
  }

  let lastImage = null
  function render() {
    const st = cur.status || 'idle'
    dot.style.background = STATUS_COLORS[st] || STATUS_COLORS.idle
    statusLabel.textContent = st
    const inImage = runtimeValueOf('image')   // 上游 load-image 的运行时输出
    if (inImage && inImage !== lastImage) { lastImage = inImage; loadThumb(inImage) }
    const latentId = (cur.outputs || {}).latent
    idLine.textContent = latentId ? `latent: ${latentId}` : ''
  }
  render()
  return {
    update(next) { cur = next || {}; render() },
    unmount() { if (blobUrl) URL.revokeObjectURL(blobUrl); el.textContent = '' },
  }
}
