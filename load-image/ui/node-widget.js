/**
 * load-image 画布组件：本地图片上传状态卡（图像对象 ID）。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 */

const STATUS_COLORS = {
  idle: '#94a3b8',
  running: '#22d3ee',
  success: '#34d399',
  failed: '#fb7185',
  skipped: '#64748b',
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
  ].join(';')

  const header = document.createElement('div')
  header.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:4px'
  const dot = document.createElement('span')
  dot.style.cssText = 'width:8px;height:8px;border-radius:50%;flex-shrink:0'
  const title = document.createElement('strong')
  title.textContent = '🖼️ Load Image'
  title.style.fontSize = '12px'
  const statusLabel = document.createElement('span')
  statusLabel.style.cssText = 'color:rgba(255,255,255,0.35);margin-left:auto'
  header.append(dot, title, statusLabel)

  const pathLine = document.createElement('div')
  pathLine.style.cssText = 'color:rgba(255,255,255,0.45);word-break:break-all'
  const imageLine = document.createElement('div')
  imageLine.style.cssText = 'color:rgba(255,255,255,0.45);word-break:break-all'

  el.append(header, pathLine, imageLine)

  function render(p) {
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    const path = (p.params || {}).image_path || ''
    pathLine.textContent = path ? `path: ${path}` : '未设置图片路径'
    const o = p.outputs || {}
    imageLine.textContent = o.image ? `image: ${o.image}` : (p.status === 'running' ? '上传中…' : '等待执行…')
  }

  render(props)

  return {
    update(next) { render(next) },
    unmount() { el.textContent = '' },
  }
}
