/**
 * widget-base.js：专属节点画布组件公共件（算子专属节点化基建）。
 *
 * 不直接使用——由 _tools/build-widget.py 与节点 ui/widget-def.js 拼接生成
 * 最终单文件 ui/node-widget.js（widget 分发要求单文件、无运行时依赖）。
 *
 * 生成产物契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 *   props.params — 当前 config.params 绑定（模板值只读展示）
 *   props.paramSources — 参数绑定来源标注：workflow=流水线参数(含当前值) /
 *     node=上游节点(含显示名与运行时值) / literal=字面值
 *   props.onParamsChange(params) — 全量写回该节点参数（回放态缺省 = 控件只读）
 *   props.preview — 预览帧 { url, progress }：执行中过程帧 + 完成后结果图
 *   props.outputs — 节点输出（含 __ 前缀内部元数据，显示时过滤）
 *   props.execution — 当前执行实例（中断/重放/对比图代理端点用）
 *
 * 节点定义（widget-def.js 提供 const WIDGET_SPEC = {...}）：
 *   icon/title    节点标题
 *   fields[]      参数控件，按序渲染：
 *     { key, label, kind:'text', mono?, placeholder? }      文本（{{ }} 绑定自动切来源标注控件）
 *     { key, label, kind:'slider', min, max, step, default } 滑块（写回字符串，执行器侧 cast）
 *     { key, label, kind:'check', default }                  复选框（写回 'true'/'false'）
 *     { key, label, kind:'select', options:[], default }     固定下拉
 *     { key, label, kind:'curves', default }                 曲线编辑器（G5 image-curves 用）：
 *       画布内拖拽控制点编辑 "x,y;x,y;..." 曲线串（x=输入亮度/y=输出亮度，0~255，
 *       按 x 升序分段线性），支持空白双击加点、双击/右键点删除、↺对角复位、⇅反相；
 *       绑定态（{{ }}）自动只读
 *     { key, label, kind:'model', modelType }                模型名下拉（经通用代理
 *       POST /api/v1/service-proxy 调第三方服务 GET /models/files，按 kind 过滤；
 *       API 路径语义由本节点生态自持）。基地址优先取同节点的 service_url_host
 *       （Studio 宿主机可达，docker 节点的 service_url 是容器内网地址取不到），
 *       未配置时回退 service_url
 *     任意 field 可加 advanced:true → 收进默认折叠的「⚙ 连接/高级」区
 *     kind:'file' 可加 noPreview:true → 不出图/视频预览（上传/拖入/路径说明都保留）。
 *       图上已有更合适的展示位时用，避免同一张图在节点上重复两遍（mask-paint 反馈：
 *       底图已在蒙版控件里显示，下面再贴一张预览纯属噪音）
 *     （service_url/service_token 等连接参数建议 advanced，保持节点紧凑）
 *   pointSelect   （可选）{pointsKey, boxKey, baseImage}：点/框选画布编辑器
 *                 （点=正样本、Shift点=负样本、拖拽=框选；原图坐标写回参数）
 *   maskMontage   （可选）{keepKey, montageKey, selectedKey}：编号网格点选保留/排除
 *                 （格子坐标用算子回传的 montage_cols/montage_tile 换算）
 *   boxList       （可选）{labelsKey, scoresKey, boxesKey}：检测框+置信度列表
 *   tileGridOverlay（可选）{gridKey}：切片网格覆盖层 + 块数/行列/预估读数
 *   compareInput  （可选）'image' 等输入键名：完成后显示前后对比滑块
 *                 （点击可弹放大对比窗）。原图两条链：① 优先 props.media.input —— 
 *                 节点执行时经 FLOWX_MEDIA{kind:"input"} 物化到 Studio 媒体缓存的
 *                 输入原图（推理服务对象是内存态，GC/重启/TTL 后必 404，物化产物
 *                 才持久可读）；② 降级：outputs.__inputs_resolved 解析对象 id，
 *                 经节点级通用代理 GET /images/{id} 实时拉取。结果图 = 预览区最终帧。
 *                 原图缺失时提示并退化为单图弹窗（不显示空壳分割线）
 *   mediaImageParam  （可选）参数键名（如 'image_path'）：直读本地图片显示
 *                    （Studio 同源媒体端点 /api/v1/media/file，点击进可缩放 lightbox）；
 *                    绑定态（{{ }}）显示占位提示
 *   mediaImageOutput （可选）输出键名（如 'file_path'）：显示落盘结果图
 *                    （同上媒体端点 + lightbox）
 *   mediaVideoOutput （可选）输出键名（如 'file_path'）：内嵌 <video> 播放
 *                    落盘视频 + ⛶ 放大 lightbox（含全屏按钮）
 *   paint         （可选）蒙版手绘：{ strokesKey, imageInput, imagePathKey?, strokesRefKey?,
 *                   strokesSubdir? }。
 *                   节点上显示底图+红罩缩略，点「✏️ 编辑蒙版」弹全屏编辑器
 *                   （笔刷/矩形/橡皮/撤销/清除，捏合缩放平移）。笔画以矢量存
 *                   strokesKey 参数（{strokes:[...], image_size:[w,h]}，分辨率
 *                   无关，可再编辑）。保存时自动抽稀（RDP + 取整，实测 5.5x 减体积）。
 *                   **外置存储（推荐）**：给了 strokesRefKey 就改存本地 JSON 文件——
 *                   保存时 POST /api/v1/media/upload?subdir=<strokesSubdir>（默认
 *                   strokes）拿回路径写进 strokesRefKey，并清空内联 strokesKey；
 *                   读取走 GET /api/v1/media/file（异步）；读回来是 404（磁盘上本来
 *                   就没这个文件，如流水线参数指向一个还没人画过的路径）视为「尚无
 *                   笔画」允许起手，其他失败才禁止编辑——避免用空笔画覆盖原内容。
 *                   适合笔画体量大、不想把编辑态文本
 *                   写进 workflow YAML / 执行快照 / SSE 的节点。
 *                   底图三条链：① 回放/执行态 __inputs_resolved
 *                   经节点级代理拉取（同 compareInput 链路）；② 编辑态从参数解析
 *                   出对象 id 走通用代理；③ imagePathKey 指定的本地路径参数直读
 *                   （Studio 媒体端点，未经执行也能衬图绘制）。三条链都无底图时
 *                   按 512x512 空白参照绘制。imagePathKey 是节点自己声明加载
 *                   本地图的参数键（如 'image_path'）。
 *                   ⚠️ 参数值一律经 resolveParam 解析：模板节点普遍写
 *                   image_path/strokes_ref: "{{ Param.xxx }}"，props.params 下发的是
 *                   模板原文、真实值只在 paramSources.paramValue —— 直读 cur.params
 *                   会把绑流水线参数误判成「绑上游」而整块丢弃（wf181 事故：
 *                   编辑器打开没有底图）。同理保存笔画时只能改节点参数 ⇒ 会把绑定
 *                   改成直填路径，提示行会明说（与 kind:'file' 上传控件同口径）。
 *                   strokesKey 对应的参数不要放进 fields（由本控件管理）。
 *   note          （可选）输出说明行
 *
 * 布局顺序（固定，图像预览置顶）：状态行 → 过程/结果预览帧 → 前后对比 →
 *   蒙版手绘 → 本地图片/视频（含路径说明行）→ 参数控件 →
 *   输出。未启用的块 display:none 不占位。
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
const BTN_CSS = 'font-size:10px;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.15);border-radius:6px;padding:2px 8px;color:rgba(255,255,255,0.75);cursor:pointer'
const WIRED_INPUT_CSS = 'width:100%;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);border-radius:6px;padding:4px 7px;font-size:10px;color:rgba(165,180,252,0.95);outline:none;box-sizing:border-box;font-family:ui-monospace,Menlo,monospace'

function h(tag, style, text) {
  const n = document.createElement(tag)
  if (style) n.style.cssText = style
  if (text !== undefined) n.textContent = text
  return n
}

const isWired = (v) => typeof v === 'string' && v.indexOf('{{') >= 0

// 移动端窄屏判定：与 Studio 的 useIsMobile 同断点（768px）。手机上节点宽度仅
// ~300px，提示词这类长文本在内联小框里无法编辑，点击后走全屏弹层放大编辑。
function isMobileViewport() {
  try {
    return window.matchMedia('(max-width: 768px)').matches
  } catch (_) {
    return false
  }
}

// ---- 笔画抽稀（mask-paint 等 paint 控件的存储瘦身）----
// 手绘时 pointermove 每帧都会 push 一个点（一笔可达数千点，实测 6 笔 7476 点 =
// 104KB 文本），而笔画在局部近似直线，绝大多数点是冗余的。保存前用
// Ramer–Douglas–Peucker 抽稀 + 坐标取整，实测降到 18.8KB（5.5x）且肉眼无差。
// ε 按笔刷直径自适应（小笔刷用更小容差，避免粗笔看得出的折线偏差）。
const STROKE_EPS_MIN = 0.75
const STROKE_EPS_MAX = 1.5
function strokeEps(st) {
  const size = Number(st && st.size) || 40
  return Math.min(STROKE_EPS_MAX, Math.max(STROKE_EPS_MIN, size / 24))
}

// simplifyPoints：RDP 折线抽稀（保留首尾；容差 eps 为图像像素）
function simplifyPoints(pts, eps) {
  // 先去掉完全重复的相邻点（pointermove 会因抖动产生）
  const src = []
  for (const p of pts || []) {
    const x = Number(p[0]), y = Number(p[1])
    if (!isFinite(x) || !isFinite(y)) continue
    const last = src[src.length - 1]
    if (last && last[0] === x && last[1] === y) continue
    src.push([x, y])
  }
  if (src.length <= 2) return src.map((p) => [Math.round(p[0]), Math.round(p[1])])
  const keep = new Array(src.length).fill(false)
  keep[0] = keep[src.length - 1] = true
  const stack = [[0, src.length - 1]]
  while (stack.length) {
    const [i, j] = stack.pop()
    if (j <= i + 1) continue
    const [x1, y1] = src[i]
    const [x2, y2] = src[j]
    const dx = x2 - x1, dy = y2 - y1
    const len = Math.hypot(dx, dy) || 1e-9
    let best = -1, bestIdx = -1
    for (let k = i + 1; k < j; k++) {
      const [x0, y0] = src[k]
      const dist = Math.abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / len
      if (dist > best) { best = dist; bestIdx = k }
    }
    if (best > eps) {
      keep[bestIdx] = true
      stack.push([i, bestIdx], [bestIdx, j])
    }
  }
  const out = []
  for (let i = 0; i < src.length; i++) {
    if (keep[i]) out.push([Math.round(src[i][0]), Math.round(src[i][1])])
  }
  return out
}

// simplifyStrokes：对 {type,size,points} 笔画数组抽稀；rect 与未知类型原样保留。
function simplifyStrokes(strokes) {
  const out = []
  for (const st of strokes || []) {
    if (!st || typeof st !== 'object') continue
    if (st.type === 'rect') {
      out.push({
        ...st,
        x: Math.round(Number(st.x) || 0), y: Math.round(Number(st.y) || 0),
        w: Math.round(Number(st.w) || 0), h: Math.round(Number(st.h) || 0),
      })
      continue
    }
    if (st.type === 'brush' || st.type === 'erase') {
      const pts = simplifyPoints(st.points, strokeEps(st))
      if (!pts.length) continue
      out.push({ ...st, points: pts })
      continue
    }
    out.push(st)
  }
  return out
}

// Studio 本地媒体文件端点（同源，浏览器自动携带认证）；读取 FlowX 所在机器
// 的本地文件（目录白名单由 Studio 配置），用于 load-image 源图预览与
// save-image/save-video 落盘结果展示
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

// 视频放大播放 lightbox：大尺寸 <video> + 显式全屏按钮（requestFullscreen，
// iOS Safari 回退 webkitEnterFullscreen；原生控制条也自带全屏入口）。
// startTime 同步自内嵌播放器；onClose 回调负责进度回同步。返回 { overlay, video }。
function buildVideoLightbox(src, caption, startTime, onClose) {
  const overlay = h('div', [
    'position:fixed', 'inset:0', 'z-index:9999',
    'background:rgba(0,0,0,0.88)', 'backdrop-filter:blur(4px)',
    'display:flex', 'align-items:center', 'justify-content:center',
  ].join(';'))
  const big = h('video', [
    'max-width:94vw', 'max-height:86vh', 'background:#000',
    'border-radius:8px', 'box-shadow:0 8px 40px rgba(0,0,0,0.6)',
  ].join(';'))
  big.controls = true
  big.loop = true
  big.playsInline = true
  big.src = src
  big.addEventListener('click', (e) => e.stopPropagation())

  const LB_BTN_CSS = [
    'position:absolute', 'width:34px', 'height:34px',
    'border-radius:50%', 'border:1px solid rgba(255,255,255,0.25)',
    'background:rgba(255,255,255,0.10)', 'color:rgba(255,255,255,0.85)',
    'font-size:16px', 'line-height:1', 'cursor:pointer', 'z-index:2',
  ].join(';')
  const closeBtn = h('button', LB_BTN_CSS + ';top:14px;right:16px', '✕')
  closeBtn.title = '关闭（Esc）'
  closeBtn.addEventListener('click', (e) => { e.stopPropagation(); onClose && onClose() })
  const fsBtn = h('button', LB_BTN_CSS + ';top:14px;right:58px', '⛶')
  fsBtn.title = '全屏播放'
  fsBtn.addEventListener('click', (e) => {
    e.stopPropagation()
    if (big.requestFullscreen) big.requestFullscreen().catch(() => {})
    else if (big.webkitEnterFullscreen) big.webkitEnterFullscreen() // iOS Safari
    else if (overlay.requestFullscreen) overlay.requestFullscreen().catch(() => {})
  })
  overlay.append(big, closeBtn, fsBtn)
  if (caption) {
    overlay.append(h('div', [
      'position:absolute', 'left:0', 'right:0', 'bottom:10px', 'text-align:center',
      'font-size:10px', 'color:rgba(255,255,255,0.45)', 'pointer-events:none',
      'padding:0 16px', 'word-break:break-all',
    ].join(';'), caption))
  }
  overlay.addEventListener('click', () => onClose && onClose())
  document.body.appendChild(overlay)
  try { big.currentTime = startTime || 0 } catch (_) {}
  big.play().catch(() => {})
  return { overlay, video: big }
}

function createNodeWidget(spec) {
  return function mount(el, props) {
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
    // nowheel/nodrag：节点内容滚动与内部交互不被画布缩放/拖拽拦截（回放态只读时控件
    // 已 disabled，但滚轮滑动长节点内容不应触发画布缩放、点图片应能放大）。
    el.classList.add('nowheel', 'nodrag')

    let cur = props
    const refreshers = []
    const editable = () => typeof cur.onParamsChange === 'function'
    const setParam = (key, value) => {
      if (!editable()) return
      const params = { ...(cur.params || {}), [key]: String(value) }
      cur = { ...cur, params }
      cur.onParamsChange(params)
    }
    // setParams：一次写回多个参数（全量替换后统一提交，避免分次写回产生中间态，
    // 如 mask-paint 保存笔画时要同时写 strokes_ref 并清空旧的内联 strokes_json）
    const setParams = (patch) => {
      if (!editable()) return
      const params = { ...(cur.params || {}) }
      for (const k of Object.keys(patch)) params[k] = String(patch[k])
      cur = { ...cur, params }
      cur.onParamsChange(params)
    }

    // ---- 放大弹层（lightbox）生命周期：全节点共用一个遮罩 ----------------
    // 图片/视频预览都经 setLightbox 打开，同一时刻最多一个弹层；关闭途径有三条
    // ——✕ 按钮、点遮罩空白、Esc——所以 onClose 必须绑上 closeLightbox，且 Esc
    // 监听常驻注册（不依赖 spec.mediaImageParam：kind:'file' 控件也有预览）。
    let lightboxEl = null
    // __cleanup：弹层携带的清理回调（如对比窗的 window 拖拽监听），关闭时一并移除防泄漏
    const closeLightbox = () => { if (lightboxEl) { if (lightboxEl.__cleanup) lightboxEl.__cleanup(); lightboxEl.remove(); lightboxEl = null } }
    const setLightbox = (el) => { closeLightbox(); lightboxEl = el }

    // ---- 长文本全屏编辑弹层（移动端）：内联 textarea 点击后放大编辑 ----
    // 弹层挂 document.body：节点在 react-flow 的 transform 容器里，fixed 会相对
    // 该容器定位，必须脱离出去才是真全屏。输入逐字符实时写回参数（与内联框
    // change 语义一致），关闭即已生效——不提供「取消还原」，避免手机上误触
    // 返回键丢内容。字号 16px：iOS 聚焦 <16px 会强制放大整页。
    let textEditorEl = null
    const closeTextEditor = () => { if (textEditorEl) { textEditorEl.remove(); textEditorEl = null } }
    const openTextEditor = (opts) => {
      const o = opts || {}
      closeTextEditor()
      const overlay = h('div', [
        'position:fixed', 'inset:0', 'z-index:10000',
        'background:rgba(6,8,15,0.97)', 'backdrop-filter:blur(6px)',
        'display:flex', 'flex-direction:column', 'box-sizing:border-box',
        'padding:calc(10px + env(safe-area-inset-top)) 10px calc(10px + env(safe-area-inset-bottom))',
      ].join(';'))
      const bar = h('div', 'display:flex;align-items:center;gap:8px;padding:0 2px 8px')
      const label = h('span', 'font-size:11px;color:rgba(255,255,255,0.6)', o.title || '')
      const count = h('span', 'font-size:10px;color:rgba(255,255,255,0.3)',
        String((o.value || '').length))
      const done = h('button', [
        'margin-left:auto', 'font-size:12px', 'font-weight:500', 'cursor:pointer',
        'background:rgba(99,102,241,0.9)', 'border:none', 'border-radius:6px',
        'padding:6px 14px', 'color:#fff',
      ].join(';'), '完成')
      done.type = 'button'
      const ta = h('textarea', [
        'flex:1', 'width:100%', 'box-sizing:border-box', 'resize:none',
        'background:rgba(255,255,255,0.06)', 'border:1px solid rgba(255,255,255,0.16)',
        'border-radius:8px', 'padding:10px', 'outline:none',
        'font-size:16px', 'line-height:1.6', 'color:rgba(255,255,255,0.92)',
        'font-family:inherit',
      ].join(';'))
      ta.value = o.value || ''
      ta.placeholder = o.placeholder || ''
      ta.spellcheck = false
      ta.addEventListener('input', () => {
        count.textContent = String(ta.value.length)
        if (typeof o.onInput === 'function') o.onInput(ta.value)
      })
      done.addEventListener('click', closeTextEditor)
      // Esc 关闭：监听挂在 textarea 上（焦点在其中），不污染全局 keydown
      ta.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') { e.stopPropagation(); closeTextEditor() }
      })
      bar.append(label, count, done)
      overlay.append(bar, ta)
      document.body.appendChild(overlay)
      textEditorEl = overlay
      setTimeout(() => { try { ta.focus() } catch (_) {} }, 0)
      return overlay
    }
    const onEsc = (e) => { if (e.key === 'Escape') closeLightbox() }
    document.addEventListener('keydown', onEsc)
    // 对比放大弹层：原图 vs 结果，可拖分割线（compareInput 区单击进入；编辑/回放态均可）。
    // 原图用 clip-path 裁切（与节点内对比区同款方案，见下）；分割线拖拽的 window 监听
    // 挂到 overlay.__cleanup，随关闭移除。
    const showCompareLightbox = (beforeUrl, afterUrl) => {
      const overlay = h('div', 'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.9);display:flex;align-items:center;justify-content:center;cursor:zoom-out')
      const box = h('div', 'position:relative;cursor:default;line-height:0')
      const afterImg = h('img', 'max-width:92vw;max-height:86vh;object-fit:contain;display:block;pointer-events:none')
      afterImg.src = afterUrl
      const tagR = h('span', 'position:absolute;right:6px;top:6px;font-size:11px;line-height:1.4;background:rgba(0,0,0,0.6);border-radius:4px;padding:1px 7px;color:rgba(255,255,255,0.9);pointer-events:none', '结果')
      box.append(afterImg, tagR)
      // 原图不可用（上游对象已驱逐且未物化）时退化为单图放大，不给空壳分割线
      if (!beforeUrl) {
        overlay.append(box)
        overlay.addEventListener('click', (e) => { if (e.target === overlay) closeLightbox() })
        setLightbox(overlay)
        document.body.append(overlay)
        return
      }
      const beforeImg = h('img', 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:block;pointer-events:none')
      beforeImg.src = beforeUrl
      const divider = h('div', 'position:absolute;top:0;bottom:0;width:2px;background:#22d3ee;cursor:ew-resize;box-shadow:0 0 6px rgba(34,211,238,0.8)')
      const knob = h('div', 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:22px;height:22px;border-radius:50%;background:#22d3ee;color:#0b1220;font-size:12px;display:flex;align-items:center;justify-content:center;pointer-events:none', '⇄')
      divider.append(knob)
      const tagL = h('span', 'position:absolute;left:6px;top:6px;font-size:11px;line-height:1.4;background:rgba(0,0,0,0.6);border-radius:4px;padding:1px 7px;color:rgba(255,255,255,0.9);pointer-events:none', '原图')
      box.insertBefore(beforeImg, tagR)
      box.insertBefore(divider, tagR)
      box.insertBefore(tagL, tagR)
      let ratio = 0.5, dragging = false
      const apply = () => {
        beforeImg.style.clipPath = `inset(0 ${(100 - ratio * 100).toFixed(2)}% 0 0)`
        divider.style.left = `calc(${(ratio * 100).toFixed(2)}% - 1px)`
      }
      apply()
      const move = (e) => {
        const rect = box.getBoundingClientRect()
        if (!rect.width) return
        const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left
        ratio = Math.min(0.98, Math.max(0.02, x / rect.width))
        apply()
      }
      divider.addEventListener('mousedown', (e) => { dragging = true; e.preventDefault(); e.stopPropagation() })
      const onMove = (e) => { if (dragging) move(e) }
      const onUp = () => { dragging = false }
      window.addEventListener('mousemove', onMove)
      window.addEventListener('mouseup', onUp)
      box.addEventListener('click', move)
      overlay.__cleanup = () => { window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp) }
      overlay.append(box)
      overlay.addEventListener('click', (e) => { if (e.target === overlay) closeLightbox() })
      setLightbox(overlay)
      document.body.append(overlay)
    }
    // 参数当前值：字面值优先，绑定值从 paramSources 取运行时/当前值（模型下拉等
    // 需要真实 service_url/token 发请求时用）
    const resolveParam = (key) => {
      const raw = (cur.params || {})[key]
      if (raw !== undefined && !isWired(raw)) return raw
      const src = (cur.paramSources || {})[key]
      if (src && src.kind === 'workflow' && src.paramValue !== undefined) return src.paramValue
      if (src && src.kind === 'node' && src.runtimeValue !== undefined) return src.runtimeValue
      return raw
    }

    // ---- 绑定参数控件（{{ ... }}）：来源标签 + 可编辑文本框 ----
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

    const textControl = (f) => {
      const raw = (cur.params || {})[f.key]
      if (isWired(raw)) return wiredControl(f.key, raw)
      const inp = h('input', INPUT_CSS + (f.mono ? ';font-family:ui-monospace,Menlo,monospace' : ''))
      inp.type = 'text'
      inp.value = raw !== undefined ? raw : (f.default !== undefined ? String(f.default) : '')
      inp.placeholder = f.placeholder || ''
      inp.addEventListener('change', () => setParam(f.key, inp.value))
      refreshers.push(() => {
        inp.disabled = !editable()
        inp.style.opacity = editable() ? '1' : '0.55'
        if (document.activeElement === inp) return
        const nv = (cur.params || {})[f.key]
        inp.value = nv !== undefined ? nv : (f.default !== undefined ? String(f.default) : '')
      })
      return inp
    }

    // ---- 本地文件控件（kind:'file'）：路径框 + 上传 + 拖拽投放 + 就地预览 ----
    // 上传经 Studio 媒体端点落盘到 FlowX 机器的 ~/flowx-input/uploads/，拿回本机
    // 路径写回参数（执行器就在本机 ⇒ 加载类节点直接读盘）。处理类节点（吃上游
    // 对象 ID）不声明 kind:'file'，界面上不会出现任何上传入口。
    const VIDEO_EXT_RE = /\.(mp4|webm|mov|mkv|avi)$/i
    const fileControl = (f) => {
      const raw = (cur.params || {})[f.key]
      const psrc = (cur.paramSources || {})[f.key] || {}
      // 绑定值分两类（props.paramSources 区分）：
      //  · kind='node'（或来源未知）＝绑上游节点输出 ⇒ 值由上游运行时决定，只读；
      //  · kind='workflow'＝绑**流水线参数** ⇒ 仍给上传/直填入口：上传即把该参数从绑定
      //    改为直填字面量（画布只有 onParamsChange 改节点参数，没有改 Param 的钩子）。
      //    为什么必须这样做：模板节点普遍写 image_path: "{{ Param.xxx }}"，
      //    若一律只读，用户在图里既不能上传也看不到预览（wf157 反馈）。
      const boundToParam = isWired(raw) && psrc.kind === 'workflow'
      if (isWired(raw) && !boundToParam) return wiredControl(f.key, raw)
      // noPreview：本字段不单独出图（图已在别处展示，如 paint 控件的底图区）；
      // 上传/拖入/路径说明照常。
      const noPreview = !!f.noPreview
      const wrap = h('div', '')
      const inp = h('input', INPUT_CSS + ';font-family:ui-monospace,Menlo,monospace')
      inp.type = 'text'
      inp.spellcheck = false
      inp.placeholder = f.placeholder || '~/flowx-input/xxx.png'
      inp.addEventListener('change', () => setParam(f.key, inp.value))

      const bar = h('div', 'display:flex;align-items:center;gap:5px;margin-top:3px')
      const pick = h('button', BTN_CSS + ';color:#a5b4fc;border-color:rgba(165,180,252,0.4)', '📁 选择文件')
      pick.title = '从本机选择文件上传到 FlowX（也可直接把文件拖到这里）'
      const status = h('span', 'font-size:9px;color:rgba(255,255,255,0.45);flex:1;line-height:1.3;word-break:break-all')
      const picker = h('input', 'display:none')
      picker.type = 'file'
      if (f.accept) picker.accept = f.accept
      bar.append(pick, status)

      const img = h('img', 'display:none;margin-top:4px;max-width:100%;max-height:120px;border-radius:6px;cursor:zoom-in;background:rgba(0,0,0,0.3)')
      const vid = h('video', 'display:none;margin-top:4px;max-width:100%;max-height:120px;border-radius:6px;background:#000')
      vid.controls = true
      vid.muted = true
      vid.playsInline = true

      // 有效路径：字面量直接用；绑流水线参数时用参数的当前值（paramValue）
      const effPath = () => {
        const v = (cur.params || {})[f.key] || ''
        if (!isWired(v)) return v
        const s2 = (cur.paramSources || {})[f.key] || {}
        return s2.kind === 'workflow' ? (s2.paramValue || '') : ''
      }
      const showPreview = () => {
        const p = effPath()
        const none = noPreview || !p || isWired(p)
        const wantVid = !none && VIDEO_EXT_RE.test(String(p))
        img.style.display = 'none'
        vid.style.display = 'none'
        if (none) return
        if (wantVid) {
          if (vid.dataset.path !== p) { vid.src = mediaUrl(p); vid.dataset.path = p }
          vid.style.display = 'block'
          return
        }
        if (img.dataset.path !== p) { img.src = mediaUrl(p); img.dataset.path = p }
        img.style.display = 'block'
      }
      img.addEventListener('click', () => {
        if (img.dataset.path) {
          setLightbox(buildImageLightbox(mediaUrl(img.dataset.path), img.dataset.path, closeLightbox))
        }
      })
      // 路径越出白名单/文件不存在：静默隐藏预览（参数本身仍可手填）
      img.addEventListener('error', () => { img.style.display = 'none' })
      vid.addEventListener('error', () => { vid.style.display = 'none' })

      const bindHint = h('div', 'font-size:9px;color:rgba(251,191,36,0.85);line-height:1.35;margin-top:3px')
      if (boundToParam) {
        bindHint.textContent = '↳ 来自流水线参数 ' + (psrc.paramName || '?')
          + '（上传或直接填写将解除绑定，改为本节点直填路径）'
      }

      let uploadBusy = false
      const setPickState = (busy) => {
        uploadBusy = busy
        pick.disabled = busy || !editable()
        pick.textContent = busy ? '⏳ 上传中…' : '📁 选择文件'
        pick.style.opacity = editable() ? '1' : '0.5'
      }
      const doUpload = (file) => {
        if (!file) return
        if (!editable()) { status.textContent = '回放态只读，无法写回参数'; return }
        const fd = new FormData()
        fd.append('file', file)
        setPickState(true)
        status.textContent = `上传 ${file.name}（${(file.size / 1048576).toFixed(1)}MB）…`
        const xhr = new XMLHttpRequest()
        xhr.open('POST', '/api/v1/media/upload')
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            status.textContent = `上传中 ${Math.round((e.loaded / e.total) * 100)}%`
          }
        }
        const fail = (msg) => { setPickState(false); status.textContent = '❌ ' + msg }
        xhr.onload = () => {
          setPickState(false)
          let body = null
          try { body = JSON.parse(xhr.responseText) } catch (_) { body = null }
          const path = body && body.data && body.data.path
          if (xhr.status >= 200 && xhr.status < 300 && path) {
            setParam(f.key, path)   // 上传物即参数值（本机路径），执行器直接读盘
            status.textContent = '✅ ' + (body.data.name || path)
              + (boundToParam ? '（已解除绑定，改为直填）' : '')
            showPreview()
          } else {
            fail((body && body.message) || ('HTTP ' + xhr.status))
          }
        }
        xhr.onerror = () => fail('网络错误')
        xhr.send(fd)
      }
      pick.addEventListener('click', () => { if (editable()) picker.click() })
      picker.addEventListener('change', () => { doUpload(picker.files && picker.files[0]); picker.value = '' })
      wrap.addEventListener('dragover', (e) => {
        e.preventDefault()
        if (editable()) wrap.style.outline = '1px dashed rgba(165,180,252,0.7)'
      })
      wrap.addEventListener('dragleave', () => { wrap.style.outline = '' })
      wrap.addEventListener('drop', (e) => {
        e.preventDefault()
        wrap.style.outline = ''
        const fl = e.dataTransfer && e.dataTransfer.files
        if (fl && fl.length) doUpload(fl[0])
      })

      refreshers.push(() => {
        inp.disabled = !editable()
        inp.style.opacity = editable() ? '1' : '0.55'
        if (document.activeElement !== inp) {
          inp.value = boundToParam
            ? (((cur.paramSources || {})[f.key] || {}).paramValue || '')
            : ((cur.params || {})[f.key] || '')
        }
        // 只同步可写性（回放态禁用），不覆盖上传中的按钮文案
        pick.disabled = uploadBusy || !editable()
        pick.style.opacity = editable() ? '1' : '0.5'
        showPreview()
      })
      wrap.append(inp)
      if (boundToParam) wrap.append(bindHint)
      wrap.append(bar, picker, img, vid)
      showPreview()
      return wrap
    }

    const readParam = (key, dflt) => {
      const v = (cur.params || {})[key]
      return v !== undefined && v !== '' ? v : dflt
    }

    const sliderControl = (f) => {
      const wrap = h('div', '')
      const val = h('span', 'color:rgba(255,255,255,0.7);font-family:ui-monospace,Menlo,monospace', String(readParam(f.key, f.default)))
      const lab = h('label', LABEL_CSS + ';display:flex;justify-content:space-between', '')
      lab.append(document.createTextNode(f.label), val)
      const rng = h('input', 'width:100%;accent-color:#22d3ee')
      rng.type = 'range'; rng.min = f.min; rng.max = f.max; rng.step = f.step
      rng.value = readParam(f.key, f.default)
      rng.disabled = !editable()
      rng.addEventListener('input', () => { val.textContent = rng.value })
      rng.addEventListener('change', () => setParam(f.key, rng.value))
      refreshers.push(() => {
        rng.disabled = !editable()
        const nv = readParam(f.key, f.default)
        if (document.activeElement !== rng) { rng.value = nv; val.textContent = String(nv) }
      })
      wrap.append(lab, rng)
      return wrap
    }

    const checkControl = (f) => {
      const wrap = h('label', 'display:flex;align-items:center;gap:6px;font-size:10px;color:rgba(255,255,255,0.75);cursor:pointer')
      const cb = h('input', 'accent-color:#22d3ee')
      cb.type = 'checkbox'
      const parse = (v) => v === true || v === 'true' || v === '1' || v === 1
      cb.checked = parse(readParam(f.key, f.default))
      cb.disabled = !editable()
      cb.addEventListener('change', () => setParam(f.key, cb.checked ? 'true' : 'false'))
      wrap.append(cb, document.createTextNode(f.label))
      refreshers.push(() => {
        cb.disabled = !editable()
        cb.checked = parse(readParam(f.key, f.default))
      })
      return wrap
    }

    const selectControl = (f) => {
      const sel = h('select', INPUT_CSS)
      for (const opt of f.options) {
        const o = h('option', '', String(opt))
        o.value = String(opt)
        sel.append(o)
      }
      sel.value = String(readParam(f.key, f.default))
      sel.addEventListener('change', () => setParam(f.key, sel.value))
      refreshers.push(() => {
        sel.disabled = !editable()
        sel.style.opacity = editable() ? '1' : '0.55'
        sel.value = String(readParam(f.key, f.default))
      })
      return sel
    }

    // 模型名下拉：经 Studio 代理拉推理端 /models/files，按 modelType 过滤；
    // 首次挂载自动加载，⟳ 手动刷新；加载失败显示原因并回退文本输入
    const modelControl = (f) => {
      const wrap = h('div', '')
      const row = h('div', 'display:flex;gap:4px;align-items:center')
      const sel = h('select', INPUT_CSS + ';flex:1')
      const refreshBtn = h('button', BTN_CSS, '⟳')
      refreshBtn.title = '刷新模型列表'
      const hint = h('div', 'font-size:9px;color:rgba(255,255,255,0.35);margin-top:2px', '模型列表加载中…')
      const curVal = () => String(readParam(f.key, f.default !== undefined ? f.default : ''))
      let loaded = false
      const setOptions = (names) => {
        sel.textContent = ''
        const cur = curVal()
        if (cur) { const o = h('option', '', cur); o.value = cur; sel.append(o) }
        for (const n of names) {
          if (n === cur) continue
          const o = h('option', '', n); o.value = n; sel.append(o)
        }
        sel.value = cur
      }
      const load = async () => {
        hint.textContent = '模型列表加载中…'
        hint.style.color = 'rgba(255,255,255,0.35)'
        // 这次查询由 Studio 宿主机发起（不是节点自己）：docker 节点的
        // service_url 是容器内网地址（如 172.17.0.1:8100），Studio 取不到，
        // 所以优先用 service_url_host（画布侧可达地址，通常绑公网隧道）
        const host = resolveParam('service_url_host')
        const url = (host && !isWired(host)) ? host : resolveParam('service_url')
        const tok = resolveParam('service_token') || ''
        if (!url || isWired(url)) {
          hint.textContent = 'service_url / service_url_host 未解析（绑定无当前值），无法拉取模型列表'
          return
        }
        try {
          const data = await serviceProxy(url, tok, '/models/files', 'GET')
          const files = (((data && data.data) || data || {}).files || []).filter((x) => x.kind === f.modelType)
          setOptions(files.map((x) => x.name))
          const via = url.replace(/^https?:\/\//, '')
          hint.textContent = files.length ? `${files.length} 个可选 · ${via}` : `无 kind=${f.modelType} 的模型 · ${via}`
          loaded = true
        } catch (e) {
          hint.textContent = '列表加载失败：' + e + '（可手动输入；docker 节点请把 service_url_host 绑成 Studio 可达地址）'
          hint.style.color = 'rgba(251,113,133,0.8)'
        }
      }
      sel.addEventListener('change', () => setParam(f.key, sel.value))
      refreshBtn.addEventListener('click', load)
      refreshers.push(() => {
        sel.disabled = !editable()
        sel.style.opacity = editable() ? '1' : '0.55'
        if (loaded) {
          const cv = curVal()
          if (sel.value !== cv) {
            const has = Array.from(sel.options).some((o) => o.value === cv)
            if (!has && cv) { const o = h('option', '', cv); o.value = cv; sel.prepend(o) }
            sel.value = cv
          }
        }
      })
      row.append(sel, refreshBtn)
      wrap.append(row, hint)
      load()
      return wrap
    }

    const textareaControl = (f) => {
      const raw = (cur.params || {})[f.key]
      if (isWired(raw)) return wiredControl(f.key, raw)
      const inp = h('textarea', INPUT_CSS + ';min-height:56px;resize:vertical;line-height:1.5')
      inp.value = raw !== undefined ? raw : (f.default !== undefined ? String(f.default) : '')
      inp.placeholder = f.placeholder || ''
      inp.spellcheck = false
      inp.addEventListener('change', () => setParam(f.key, inp.value))
      refreshers.push(() => {
        inp.disabled = !editable()
        inp.style.opacity = editable() ? '1' : '0.55'
        if (document.activeElement === inp) return
        const nv = (cur.params || {})[f.key]
        inp.value = nv !== undefined ? nv : (f.default !== undefined ? String(f.default) : '')
      })
      // 移动端：内联框退化为「摘要 + 放大入口」，点击开全屏弹层编辑长文本。
      // 只读 + pointer 光标：避免软键盘弹在 56px 高的小框里；回放态（不可编辑）
      // 点开无效，与桌面一致保持只读
      if (isMobileViewport()) {
        inp.readOnly = true
        inp.style.cursor = 'pointer'
        inp.style.paddingBottom = '16px'
        const wrap = h('div', 'position:relative')
        const hint = h('span', [
          'position:absolute', 'right:7px', 'bottom:3px', 'font-size:9px',
          'color:rgba(165,180,252,0.75)', 'pointer-events:none',
        ].join(';'), '⤢ 放大编辑')
        const expand = (e) => {
          if (e && e.preventDefault) e.preventDefault()
          if (!editable()) return
          openTextEditor({
            title: f.label || f.key,
            value: inp.value,
            placeholder: f.placeholder || '',
            onInput: (v) => { inp.value = v; setParam(f.key, v) },
          })
        }
        inp.addEventListener('click', expand)
        inp.addEventListener('focus', () => expand())
        refreshers.push(() => { hint.style.display = editable() ? 'block' : 'none' })
        wrap.append(inp, hint)
        return wrap
      }
      return inp
    }

    // ---- 曲线编辑器（kind:'curves'）：拖拽控制点编辑 "x,y;x,y;..." ----
    // 语义与 image-curves 节点一致：x=输入亮度(0~255)、y=输出亮度(0~255)，
    // 按 x 升序分段线性。交互：拖拽已有点 / 空白处双击加点 / 双击点删除 /
    // 右键点删除 / ↺ 复位。写回的是曲线串（分辨率无关，纯参数，无额外状态）。
    const CURVE_W = 236
    const CURVE_H = 140
    const parseCurve = (raw) => {
      const pts = []
      for (const part of String(raw === undefined || raw === null ? '' : raw).split(';')) {
        const m = part.split(',')
        if (m.length !== 2) continue
        const x = Number(m[0])
        const y = Number(m[1])
        if (!isFinite(x) || !isFinite(y)) continue
        pts.push([Math.min(255, Math.max(0, x)), Math.min(255, Math.max(0, y))])
      }
      pts.sort((a, b) => a[0] - b[0])
      return pts.length ? pts : [[0, 0], [255, 255]]
    }
    const fmtCurve = (pts) => pts.map((p) => Math.round(p[0]) + ',' + Math.round(p[1])).join(';')

    const curvesControl = (f) => {
      const wrap = h('div', '')
      const cv = h('canvas', 'width:100%;height:' + CURVE_H + 'px;border-radius:6px;background:rgba(0,0,0,0.35);border:1px solid rgba(255,255,255,0.12);touch-action:none;cursor:crosshair;display:block')
      const val = h('div', 'font-size:9px;color:rgba(255,255,255,0.5);font-family:ui-monospace,Menlo,monospace;margin-top:3px;word-break:break-all')
      const btns = h('div', 'display:flex;gap:4px;margin-top:4px')
      const reset = h('button', BTN_CSS, '↺ 对角复位')
      const inv = h('button', BTN_CSS, '⇅ 反相')
      const hint = h('div', 'font-size:9px;color:rgba(255,255,255,0.3);margin-top:3px',
        '拖拽控制点 · 空白双击加点 · 双击点删除')
      let pts = parseCurve(readParam(f.key, f.default))
      let drag = -1

      // 画布坐标 ↔ 曲线坐标：y 轴向上为亮（y=255 在顶部）
      const toCv = (p) => [p[0] / 255 * CURVE_W, CURVE_H - p[1] / 255 * CURVE_H]
      const fromCv = (cx, cy) => [
        Math.min(255, Math.max(0, cx / CURVE_W * 255)),
        Math.min(255, Math.max(0, 255 - cy / CURVE_H * 255)),
      ]
      const lutOf = () => {
        const lut = []
        for (let i = 0; i < 256; i++) {
          if (i <= pts[0][0]) { lut.push(pts[0][1]); continue }
          if (i >= pts[pts.length - 1][0]) { lut.push(pts[pts.length - 1][1]); continue }
          let v = i
          for (let k = 0; k < pts.length - 1; k++) {
            const [x0, y0] = pts[k]
            const [x1, y1] = pts[k + 1]
            if (x0 <= i && i <= x1) { v = x1 === x0 ? y1 : y0 + (i - x0) / (x1 - x0) * (y1 - y0); break }
          }
          lut.push(v)
        }
        return lut
      }
      const draw = () => {
        const dpr = window.devicePixelRatio || 1
        cv.width = CURVE_W * dpr
        cv.height = CURVE_H * dpr
        const g = cv.getContext('2d')
        g.setTransform(dpr, 0, 0, dpr, 0, 0)
        g.clearRect(0, 0, CURVE_W, CURVE_H)
        g.strokeStyle = 'rgba(255,255,255,0.07)'
        g.lineWidth = 1
        for (let i = 1; i < 4; i++) {
          g.beginPath(); g.moveTo(i * CURVE_W / 4, 0); g.lineTo(i * CURVE_W / 4, CURVE_H); g.stroke()
          g.beginPath(); g.moveTo(0, i * CURVE_H / 4); g.lineTo(CURVE_W, i * CURVE_H / 4); g.stroke()
        }
        g.strokeStyle = 'rgba(255,255,255,0.16)'
        g.beginPath(); g.moveTo(0, CURVE_H); g.lineTo(CURVE_W, 0); g.stroke()
        const lut = lutOf()
        g.strokeStyle = '#22d3ee'
        g.lineWidth = 1.6
        g.beginPath()
        for (let i = 0; i < 256; i++) {
          const x = i / 255 * CURVE_W
          const y = CURVE_H - Math.min(255, Math.max(0, lut[i])) / 255 * CURVE_H
          if (i === 0) g.moveTo(x, y); else g.lineTo(x, y)
        }
        g.stroke()
        pts.forEach((p) => {
          const [x, y] = toCv(p)
          g.fillStyle = '#0f172a'
          g.strokeStyle = '#22d3ee'
          g.lineWidth = 1.6
          g.beginPath(); g.arc(x, y, 4.5, 0, Math.PI * 2); g.fill(); g.stroke()
        })
        val.textContent = fmtCurve(pts)
      }
      const commit = () => draw()
      const save = () => { setParam(f.key, fmtCurve(pts)); refreshCurve() }
      const hitIndex = (cx, cy) => {
        for (let i = 0; i < pts.length; i++) {
          const [x, y] = toCv(pts[i])
          if ((cx - x) * (cx - x) + (cy - y) * (cy - y) <= 100) return i
        }
        return -1
      }
      // 报告上游参数变化（绑定串被外部改写 / 回放态切换）
      const refreshCurve = () => {
        const raw = (cur.params || {})[f.key]
        if (isWired(raw)) {
          // 绑定态：只读展示绑定来源，不参与编辑（与其它控件一致）
          const next = parseCurve(resolveParam(f.key) === raw ? f.default : resolveParam(f.key))
          pts = next
        } else {
          const next = parseCurve(raw !== undefined && raw !== '' ? raw : f.default)
          pts = next
        }
        const locked = isWired((cur.params || {})[f.key])
        cv.style.opacity = locked ? '0.45' : '1'
        reset.disabled = inv.disabled = !editable() || locked
        draw()
      }
      const evtPos = (e) => {
        const r = cv.getBoundingClientRect()
        return [
          Math.min(CURVE_W, Math.max(0, e.clientX - r.left)),
          Math.min(CURVE_H, Math.max(0, e.clientY - r.top)),
        ]
      }
      const move = (i, cx, cy) => {
        const [x, y] = fromCv(cx, cy)
        const lo = i === 0 ? 0 : pts[i - 1][0]
        const hi = i === pts.length - 1 ? 255 : pts[i + 1][0]
        pts[i] = [Math.min(hi, Math.max(lo, x)), y]
      }
      cv.addEventListener('pointerdown', (e) => {
        if (!editable() || isWired((cur.params || {})[f.key])) return
        const [cx, cy] = evtPos(e)
        const i = hitIndex(cx, cy)
        if (e.button === 2 || e.detail === 2 && i > 0 && i < pts.length - 1) {
          if (i > 0 && i < pts.length - 1) { pts.splice(i, 1); save() }
          return
        }
        if (i >= 0) { drag = i } else {
          const [x, y] = fromCv(cx, cy)
          pts.push([x, y])
          pts.sort((a, b) => a[0] - b[0])
          drag = pts.findIndex((p) => p[0] === x && p[1] === y)
        }
        cv.setPointerCapture(e.pointerId)
        commit()
      })
      cv.addEventListener('pointermove', (e) => {
        if (drag < 0) return
        const [cx, cy] = evtPos(e)
        move(drag, cx, cy)
        commit()
      })
      const endDrag = () => { if (drag >= 0) { drag = -1; save() } }
      cv.addEventListener('pointerup', endDrag)
      cv.addEventListener('pointercancel', endDrag)
      cv.addEventListener('contextmenu', (e) => e.preventDefault())
      reset.addEventListener('click', () => { pts = [[0, 0], [255, 255]]; save() })
      inv.addEventListener('click', () => { pts = [[0, 255], [255, 0]]; save() })
      refreshers.push(refreshCurve)
      refreshCurve()
      btns.append(reset, inv)
      wrap.append(cv, val, btns, hint)
      return wrap
    }

    const controlOf = (f) => {
      // slider 的 label 内嵌在控件里（右侧带当前值），外层不再重复加 label
      if (f.kind === 'slider') return field('', sliderControl(f))
      if (f.kind === 'check') return field('', checkControl(f))
      if (f.kind === 'select') return field(f.label, selectControl(f))
      if (f.kind === 'model') return field(f.label, modelControl(f))
      if (f.kind === 'textarea') return field(f.label, textareaControl(f))
      if (f.kind === 'curves') return field(f.label, curvesControl(f))
      if (f.kind === 'file') return field(f.label, fileControl(f))
      return field(f.label, textControl(f))
    }

    // ---- 头部 ----
    const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
    const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
    const title = h('strong', 'font-size:12px', (spec.icon || '🧩') + ' ' + (spec.title || ''))
    const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
    header.append(dot, title, statusLabel)

    // ---- 预览/结果区（过程帧 + 完成帧 + 进度条 + 中断 + 点击放大） ----
    const prevWrap = h('div', 'display:none;margin-bottom:6px')
    const prevImg = h('img', 'width:100%;border-radius:8px;display:block;cursor:zoom-in;background:rgba(0,0,0,0.3)')
    prevImg.alt = '预览'
    let overlay = null
    prevImg.addEventListener('click', () => {
      if (!prevImg.src || overlay) return
      overlay = h('div', 'position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.85);display:flex;align-items:center;justify-content:center;cursor:zoom-out')
      const big = h('img', 'max-width:95vw;max-height:95vh;object-fit:contain')
      big.src = prevImg.src
      overlay.append(big)
      overlay.addEventListener('click', () => { overlay.remove(); overlay = null })
      document.body.append(overlay)
    })
    prevImg.addEventListener('error', () => { prevWrap.style.display = 'none' })
    const progOuter = h('div', 'height:4px;background:rgba(255,255,255,0.08);border-radius:2px;margin-top:4px;overflow:hidden')
    const progInner = h('div', 'height:100%;background:#22d3ee;width:0%;transition:width .3s')
    progOuter.append(progInner)
    // ---- 设计期通用代理助手：base 由调用方显式给定（Studio 只做认证转发，
    // 不认识对端 API 形态）；第三方 API 路径语义由本节点生态自持
    const serviceProxy = async (base, tok, path, method, payload) => {
      const r = await fetch('/api/v1/service-proxy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          service_url: base, service_token: tok || '',
          method: method || 'GET', path,
          body: payload === undefined ? null : payload, timeout_sec: 30,
        }),
      })
      const text = await r.text()
      let data = null
      try { data = JSON.parse(text) } catch { /* 上游非 JSON 响应 */ }
      if (!r.ok) {
        const detail = (data && (data.message || data.detail || data.error)) || text.slice(0, 160)
        throw new Error('HTTP ' + r.status + (detail ? '：' + detail : ''))
      }
      return data
    }

    // ---- 节点级通用代理助手：第三方 API 路径语义由本节点生态自持 ----
    const nodeProxyUrl = (path, method) =>
      `/api/v1/executions/${cur.execution.id}/nodes/${encodeURIComponent(cur.nodeId)}/service-proxy?path=${encodeURIComponent(path)}&method=${method || 'GET'}`
    const resolvedInputs = () => {
      const raw = cur.outputs && cur.outputs.__inputs_resolved
      if (!raw) return null
      try { return typeof raw === 'string' ? JSON.parse(raw) : raw } catch { return null }
    }
    // 当前任务标识：优先 preview.jobId（executor-base 上报）；老节点包只在
    // preview url 里携带（{base}/preview/{job_id}），作解析回退
    const currentJobId = () => {
      const pv = cur.preview || {}
      if (pv.jobId) return pv.jobId
      const m = /\/preview\/([A-Za-z0-9_-]+)/.exec(pv.url || '')
      return m ? m[1] : ''
    }

    const prevRow = h('div', 'display:flex;align-items:center;gap:6px;margin-top:4px')
    const prevInfo = h('span', 'font-size:9px;color:rgba(255,255,255,0.4);flex:1;word-break:break-all')
    const stopBtn = h('button', BTN_CSS + ';color:#fb7185;border-color:rgba(251,113,133,0.4)', '■ 中断')
    stopBtn.title = '中断该节点在第三方服务上的运行中任务（经节点级通用代理 POST /interrupt）'
    stopBtn.addEventListener('click', async () => {
      if (!cur.execution) { prevInfo.textContent = '无执行实例'; return }
      const jobId = currentJobId()
      if (!jobId) { prevInfo.textContent = '节点未上报任务标识，无法中断'; return }
      stopBtn.disabled = true
      try {
        const r = await fetch(nodeProxyUrl('/interrupt', 'POST'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_id: jobId }),
        })
        prevInfo.textContent = r.ok ? '已请求中断' : `中断失败 HTTP ${r.status}`
      } catch (e) {
        prevInfo.textContent = '中断失败：' + e
      } finally {
        stopBtn.disabled = false
      }
    })
    prevRow.append(prevInfo, stopBtn)
    prevWrap.append(prevImg, progOuter, prevRow)

    // ---- 前后对比区（compareInput）：输入图 vs 结果帧，拖拽分割线 ----
    const cmpWrap = h('div', 'display:none;margin-bottom:6px')
    if (spec.compareInput) {
      const box = h('div', 'position:relative;width:100%;border-radius:8px;overflow:hidden;background:rgba(0,0,0,0.3);user-select:none;touch-action:none;cursor:zoom-in')
      const afterImg = h('img', 'width:100%;display:block;pointer-events:none')   // 结果（底层全显）
      // 原图与结果同幅铺满 box，clip-path 按分割线裁掉右侧——替代旧「beforeWrap 定宽 +
      // beforeImg width:100/ratio%」换算（那会被全局 img max-width:100% 顶回 beforeWrap
      // 宽度，把原图缩在左边、失去对比效果）。
      const beforeImg = h('img', 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;pointer-events:none')
      const divider = h('div', 'position:absolute;top:0;bottom:0;width:2px;background:#22d3ee;cursor:ew-resize;box-shadow:0 0 6px rgba(34,211,238,0.8)')
      const knob = h('div', 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:18px;height:18px;border-radius:50%;background:#22d3ee;color:#0b1220;font-size:10px;display:flex;align-items:center;justify-content:center;pointer-events:none', '⇄')
      divider.append(knob)
      const tagL = h('span', 'position:absolute;left:4px;top:4px;font-size:9px;background:rgba(0,0,0,0.55);border-radius:4px;padding:1px 5px;color:rgba(255,255,255,0.85);pointer-events:none', '原图')
      const tagR = h('span', 'position:absolute;right:4px;top:4px;font-size:9px;background:rgba(0,0,0,0.55);border-radius:4px;padding:1px 5px;color:rgba(255,255,255,0.85);pointer-events:none', '结果')
      box.append(afterImg, beforeImg, divider, tagL, tagR)
      const cmpHint = h('div', 'font-size:9px;color:rgba(255,255,255,0.35);margin-top:2px')
      cmpWrap.append(box, cmpHint)
      let ratio = 0.5
      const applyRatio = () => {
        beforeImg.style.clipPath = `inset(0 ${(100 - ratio * 100).toFixed(2)}% 0 0)`
        divider.style.left = `calc(${(ratio * 100)}% - 1px)`
      }
      applyRatio()
      let dragging = false
      const move = (e) => {
        const rect = box.getBoundingClientRect()
        if (!rect.width) return
        const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left
        ratio = Math.min(0.98, Math.max(0.02, x / rect.width))
        applyRatio()
      }
      divider.addEventListener('mousedown', (e) => { dragging = true; e.preventDefault(); e.stopPropagation() })
      window.addEventListener('mousemove', (e) => { if (dragging) move(e) })
      window.addEventListener('mouseup', () => { dragging = false })
      // 单击（非拖分割线）→ 弹放大对比窗；拖分割线走 divider 自身
      // 原图加载失败时不把空 src 带进弹窗（退化为单图放大）
      box.addEventListener('click', () => {
        if (afterImg.src) showCompareLightbox(cmpBeforeState === 'fail' ? '' : beforeImg.src, afterImg.src)
      })
      let cmpLoadedFor = ''
      // 原图当前用的 URL（物化产物或实时代理链）：用于换源重载
      let cmpBeforeFor = ''
      // 原图加载状态：'' 未定 / 'ok' / 'fail'；用于提示文案与弹窗降级
      let cmpBeforeState = ''
      // 提示文案只由原图状态决定。不能让 afterImg.onload 无条件改文案：原图失败时
      // 两事件完成顺序不定，会被正常文案覆盖，把失败提示吃掉（曾致「原图缩在左边」）。
      const syncHint = () => {
        cmpHint.textContent = cmpBeforeState === 'fail'
          ? '原图加载失败（对象已驱逐且未物化，仅显示结果）'
          : '拖动分割线对比 原图 / 结果（点击放大）'
      }
      beforeImg.onload = () => { cmpBeforeState = 'ok'; syncHint() }
      beforeImg.onerror = () => { cmpBeforeState = 'fail'; syncHint() }
      afterImg.onload = syncHint
      syncHint()
      refreshers.push(() => {
        // 结果帧 = 预览区最终帧（progress 1.0）；原图优先取 Studio 物化的 kind=input
        // 产物（props.media.input）：推理服务的图像对象是内存态（GC/重启/TTL 后必 404），
        // 物化产物随媒体缓存持久可读。拿不到物化产物时降级走节点级代理实时拉取。
        const pv = cur.preview
        const effectiveUrl = pv && pv.url
        const done = effectiveUrl && (typeof pv.progress !== 'number' || pv.progress >= 1)
        if (!done || !cur.execution) {
          cmpWrap.style.display = 'none'
          cmpLoadedFor = ''
          cmpBeforeFor = ''
          return
        }
        const resolved = resolvedInputs()
        const ref = resolved && resolved[spec.compareInput]
        const inId = ref && (typeof ref === 'string' ? ref : (ref.$id || ref.id || ''))
        const pinned = cur.media && cur.media.input
        if (!pinned && !inId) {
          cmpWrap.style.display = 'none'
          cmpLoadedFor = ''
          cmpBeforeFor = ''
          return
        }
        // 立即显示对比区并隐藏上方预览区：不能等 afterImg.onload——结果帧往往命中
        // 浏览器缓存（预览区已加载过同 URL），重复赋相同 src 不再触发 load 事件，
        // 会致对比区/预览区双双停在隐藏态。
        cmpWrap.style.display = 'block'
        prevWrap.style.display = 'none'
        if (cmpLoadedFor !== effectiveUrl) {
          cmpLoadedFor = effectiveUrl
          afterImg.src = effectiveUrl
        }
        const beforeUrl = pinned || nodeProxyUrl('/images/' + inId, 'GET')
        if (cmpBeforeFor !== beforeUrl) {
          cmpBeforeFor = beforeUrl
          cmpBeforeState = ''
          syncHint()
          beforeImg.src = beforeUrl
        }
      })
    }

    // ---- 蒙版手绘区（spec.paint）：节点缩略预览 + ✏️弹窗编辑器 ----
    // 笔画矢量有两个存放位：
    //   P.strokesRefKey —— 外置文件路径（推荐）：保存时经 POST /api/v1/media/upload
    //     落盘成 JSON（data 类，路径形如 ~/flowx-input/uploads/strokes/xxx.json），
    //     参数里只留短路径 ⇒ workflow YAML / 执行快照 / SSE 不会被笔画文本撞大。
    //     读取走 GET /api/v1/media/file（异步）。
    //   P.strokesKey —— 内联 JSON（旧写法，保留兼容）；外置键非空时以外置为准，
    //     保存外置后旧内联值会被清空。
    // 底图走 compareInput 同款链路（上次执行 __inputs_resolved + 节点级代理
    // GET /images/{id}）；未执行过给 512x512 空白参照系，执行器按原图分辨率缩放光栅化。
    let closePaintEditor = null
    const paintWrap = h('div', 'display:none;margin-bottom:6px')
    if (spec.paint) {
      const P = spec.paint
      const REF_KEY = P.strokesRefKey || ''
      const SUBDIR = P.strokesSubdir || 'strokes'
      let strokes = []
      let refW = 512, refH = 512
      let baseImg = null, baseFor = ''
      let thumbKey = ''
      let strokesSrc = null    // 已加载（或已尝试加载）的有效参数值（resolveParam 之后）
      let strokesErr = ''      // 读取失败信息：失败时禁止编辑，避免用空笔画覆盖原内容
      let strokesNoFile = false // 外置笔画文件 404（磁盘上本来就没有）：视为空笔画，允许起手
      let rebindDone = false   // 本次会话保存过笔画且原值是绑定（已解绑为直填路径）
      let strokesLoading = false
      const thumb = h('canvas', 'display:block;margin:0 auto;border-radius:8px;background:#0b1220;max-width:100%;max-height:260px;width:auto;height:auto;cursor:pointer')
      const bar = h('div', 'display:flex;align-items:center;gap:6px;margin-top:4px')
      const editBtn = h('button', BTN_CSS + ';color:#f472b6;border-color:rgba(244,114,182,0.4)', '✏️ 编辑蒙版')
      editBtn.title = '打开蒙版编辑器：笔刷/矩形涂抹重绘区域（白=重绘、黑=保留），橡皮修正'
      const paintHint = h('span', 'font-size:9px;color:rgba(255,255,255,0.4);flex:1;line-height:1.4')
      const paintNote = h('div', 'display:none;font-size:9px;color:rgba(251,191,36,0.85);line-height:1.35;margin-top:3px')
      bar.append(editBtn, paintHint)
      paintWrap.append(thumb, bar, paintNote)
      paintWrap.style.display = 'block'

      // 参数当前值：字面量 = 原值；绑定 = resolveParam 解析出的当前值/运行时值。
      // 必须走 resolveParam（同底图/模型下拉链路）：模板节点普遍写
      // strokes_ref: "{{ Param.xxx }}"，直接读 cur.params 拿到的是模板原文，会被
      // 下面的 isWired 当成「绑上游」整块丢弃 ⇒ 已保存的笔画永远读不回来（wf181）。
      const paramText = (key) => {
        const v = resolveParam(key)
        return v === undefined || v === null ? '' : String(v)
      }
      // 工作坐标系 = 底图真实分辨率（有底图时；与编辑器一致），否则沿用笔画存储参照
      // refW/refH。缩略图与提示行都用它 —— 否则 832×1216 的底图会被当成 512×512 方框
      // 里的内容，竖向压扁（用户实测反馈）。
      const workW = () => (baseImg ? baseImg.naturalWidth : refW)
      const workH = () => (baseImg ? baseImg.naturalHeight : refH)
      // 笔画坐标换算：把存储参照系（refW×refH）里的笔画映射到目标坐标系。x、y 可能
      // 不等比（换过不同长宽比的底图），故分开缩放、线宽取几何平均；缩略图与编辑器
      // 打开时共用这一份，避免两处算法漂移。始终返回深拷贝（编辑器会就地改写数组）。
      const mapStrokes = (src, toW, toH) => {
        const kx = toW / refW, ky = toH / refH, km = (kx + ky) / 2
        return (src || []).map((s) => {
          const o = JSON.parse(JSON.stringify(s))
          if (kx !== 1 || ky !== 1) {
            if (o.points) o.points = o.points.map((p) => [p[0] * kx, p[1] * ky])
            if (o.size) o.size = o.size * km
            if (o.type === 'rect') { o.x *= kx; o.y *= ky; o.w *= kx; o.h *= ky }
          }
          return o
        })
      }
      // 有效笔画来源：外置引用非空时优先，否则回退内联（兼容旧 workflow）
      const strokesValue = () => {
        const ref = REF_KEY ? paramText(REF_KEY) : ''
        if (ref !== '') return ref
        return paramText(P.strokesKey)
      }

      // 解析笔画文本（内联 JSON 或外置文件内容）；解析失败不改动现有 strokes
      const applyStrokesJSON = (txt) => {
        const d = JSON.parse(txt)
        strokes = Array.isArray(d.strokes) ? d.strokes : []
        const sz = d.image_size
        if (Array.isArray(sz) && Number(sz[0]) > 0 && Number(sz[1]) > 0) {
          refW = Number(sz[0]); refH = Number(sz[1])
        }
        return strokes
      }

      const syncStrokes = () => {
        const raw = strokesValue()
        if (raw === strokesSrc) return
        strokesSrc = raw
        strokesErr = ''
        strokesNoFile = false
        strokesLoading = false
        strokes = []
        if (!raw) return
        // 解析后仍是模板串 = 绑了上游节点输出（或 Param 里没有该键）：只读，不解析
        if (isWired(raw)) return
        const txt = raw.trim()
        if (txt.charAt(0) === '{') {          // 内联 JSON（旧写法）
          try { applyStrokesJSON(txt) } catch (_) { strokesErr = '内联 JSON 解析失败'; strokes = [] }
          return
        }
        // 外置文件：异步读回（studio 同源媒体端点，目录白名单由 Studio 配置）
        strokesLoading = true
        const want = raw
        fetch(mediaUrl(want), { credentials: 'same-origin' })
          .then((r) => {
            if (r.status === 404) {           // 磁盘上还没有这个笔画文件：不算错误
              const e = new Error('HTTP 404')
              e.notFound = true
              throw e
            }
            if (!r.ok) throw new Error('HTTP ' + r.status)
            return r.text()
          })
          .then((t) => {
            if (strokesSrc !== want) return    // 期间参数又变了，丢弃过期响应
            try { applyStrokesJSON(t) } catch (e) { strokes = []; strokesErr = '笔画文件解析失败' }
            strokesLoading = false
            thumbKey = ''; renderThumb(); updateHint()
          })
          .catch((e) => {
            if (strokesSrc !== want) return
            strokesLoading = false
            strokes = []
            // 404 = 该路径尚无笔画文件（首次绘制 / 流水线参数指向还没人画过的路径，
            // wf181 实测）⇒ 视为空笔画、允许起手；其他失败（网络/权限/5xx/JSON 坏）
            // 才禁止编辑，避免用空笔画覆盖原内容
            strokesNoFile = !!(e && e.notFound)
            strokesErr = strokesNoFile ? '' : '读取失败（' + String((e && e.message) || e) + '）'
            thumbKey = ''; renderThumb(); updateHint()
          })
      }

      // 保存：抽稀后落盘成外置 JSON（REF_KEY 存在）或写回内联参数
      const saveStrokes = (payload, done) => {
        if (!REF_KEY) { setParam(P.strokesKey, payload); done(null, ''); return }
        const fd = new FormData()
        fd.append('file', new Blob([payload], { type: 'application/json' }), 'strokes.json')
        const xhr = new XMLHttpRequest()
        xhr.open('POST', '/api/v1/media/upload?subdir=' + encodeURIComponent(SUBDIR))
        xhr.onload = () => {
          let body = null
          try { body = JSON.parse(xhr.responseText) } catch (_) { body = null }
          const path = body && body.data && body.data.path
          if (xhr.status >= 200 && xhr.status < 300 && path) done(null, path)
          else done((body && body.message) || ('HTTP ' + xhr.status), '')
        }
        xhr.onerror = () => done('网络错误', '')
        xhr.send(fd)
      }

      // 画一笔到 ctx（k = 画布像素 / 图像像素）；erase 用 destination-out 挖洞，
      // 调用方需保证笔画画在独立图层上（否则会擦掉底图）
      const drawStroke = (ctx, st, k) => {
        ctx.globalCompositeOperation = st.type === 'erase' ? 'destination-out' : 'source-over'
        ctx.fillStyle = 'rgba(244,63,94,1)'
        ctx.strokeStyle = 'rgba(244,63,94,1)'
        if (st.type === 'rect') {
          const x = (st.x || 0) * k, y = (st.y || 0) * k
          const w = (st.w || 0) * k, hh = (st.h || 0) * k
          ctx.fillRect(Math.min(x, x + w), Math.min(y, y + hh), Math.abs(w), Math.abs(hh))
          return
        }
        if (st.type === 'brush' || st.type === 'erase') {
          const pts = st.points || []
          if (!pts.length) return
          ctx.lineWidth = Math.max(1, (st.size || 40) * k)
          ctx.lineCap = 'round'
          ctx.lineJoin = 'round'
          ctx.beginPath()
          ctx.moveTo(pts[0][0] * k, pts[0][1] * k)
          if (pts.length === 1) ctx.lineTo(pts[0][0] * k + 0.01, pts[0][1] * k)
          else for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0] * k, pts[i][1] * k)
          ctx.stroke()
        }
      }

      const updateHint = () => {
        // 编辑按钮的可用性跟提示同步刷新：异步读回完成/失败不会触发宿主 update()，
        // 若只在 refreshers 里刷新，按钮会停在「笔画读取中」的禁用态直到下次 props 变化。
        const canEdit = editable() && !strokesLoading && !strokesErr
        editBtn.disabled = !canEdit
        editBtn.style.opacity = canEdit ? '1' : '0.5'
        editBtn.title = !editable() ? '回放态只读，无法写回参数'
          : strokesLoading ? '笔画读取中…'
            : strokesErr ? '笔画读取失败，已禁止编辑（避免覆盖原内容）：' + strokesErr
              : '打开蒙版编辑器：笔刷/矩形涂抹重绘区域（白=重绘、黑=保留），橡皮修正'
        // 解绑说明：strokes_ref 绑的是流水线参数时，画布只有 onParamsChange 改节点
        // 参数 ⇒ 保存只能改成「本节点直填路径」（与 kind:'file' 上传控件同口径）。
        // 提前告知 + 保存后确认，免得用户以为参数被改坏（wf181 连带疑问）。
        // 注意刷新要在下面的 loading/err 早退之前，否则保存完那一刻提示不更新。
        const refRaw = REF_KEY ? String((cur.params || {})[REF_KEY] || '') : ''
        const refSrc = REF_KEY ? ((cur.paramSources || {})[REF_KEY] || {}) : {}
        if (REF_KEY && isWired(refRaw) && refSrc.kind === 'workflow') {
          paintNote.textContent = '↳ 笔画来自流水线参数 ' + (refSrc.paramName || '')
            + '（保存后解除绑定，改为本节点直填路径）'
          paintNote.style.display = 'block'
        } else if (REF_KEY && rebindDone) {
          paintNote.textContent = '↳ 已解除流水线参数绑定，笔画路径改为本节点直填'
          paintNote.style.display = 'block'
        } else {
          paintNote.textContent = ''
          paintNote.style.display = 'none'
        }
        if (strokesLoading) { paintHint.textContent = '⏳ 读取笔画…'; return }
        if (strokesErr) { paintHint.textContent = '❌ ' + strokesErr; return }
        const noFileTip = strokesNoFile ? '（尚无笔画文件）' : ''
        paintHint.textContent = strokes.length
          ? `${strokes.length} 笔 · ${workW()}×${workH()}${baseImg ? '' : '（空白参照）'}${noFileTip}`
          : (baseImg
            ? `底图 ${baseImg.naturalWidth}×${baseImg.naturalHeight} · 未绘制${noFileTip}`
            : '未绘制 · 可空白起手（512×512 参照）' + noFileTip)
      }

      const maskCv = document.createElement('canvas')
      // 缩略图画布：按工作坐标系（底图）的真实长宽比出，长边封顶 384 内部像素；
      // 显示侧由 CSS max-height 收口（节点容器是固定高度，图太高会把下面控件挤出可视区）。
      // 事故：原来固定按 refW 出方框（512×512 参照）⇒ 832×1216 的底图被塞进方框，
      // 竖向压扁变形。
      const THUMB_MAX_SIDE = 384
      const renderThumb = () => {
        const ww = workW(), wh = workH()
        const s = Math.min(THUMB_MAX_SIDE / ww, THUMB_MAX_SIDE / wh)
        const W = Math.max(1, Math.round(ww * s))
        const H = Math.max(1, Math.round(wh * s))
        if (thumb.width !== W || thumb.height !== H) { thumb.width = W; thumb.height = H }
        const ctx = thumb.getContext('2d')
        ctx.fillStyle = '#0b1220'
        ctx.fillRect(0, 0, W, H)
        if (baseImg) ctx.drawImage(baseImg, 0, 0, W, H)
        if (strokes.length) {
          maskCv.width = W; maskCv.height = H
          const mctx = maskCv.getContext('2d')
          mctx.clearRect(0, 0, W, H)
          const k = W / ww
          for (const st of mapStrokes(strokes, ww, wh)) drawStroke(mctx, st, k)
          ctx.save(); ctx.globalAlpha = 0.55; ctx.drawImage(maskCv, 0, 0); ctx.restore()
        }
      }

      // ---- 全屏编辑器 ----
      const openEditor = () => {
        if (closePaintEditor) return
        if (strokesLoading || strokesErr) return   // 笔画没读回来就不进编辑器（免得空笔画覆盖）
        // 工作坐标系 = 当前底图分辨率（无底图沿用存储参照）；与存储参照不同时
        // 先把已有笔画换算过来（mapStrokes 与缩略图同一份算法），保存时以当前坐标系为准
        const W0 = workW()
        const H0 = workH()
        const wk = mapStrokes(strokes, W0, H0)
        let tool = 'brush'
        let brushSize = 40
        let curStroke = null

        const overlay = h('div', [
          'position:fixed', 'inset:0', 'z-index:9999', 'background:rgba(0,0,0,0.92)',
          'display:flex', 'flex-direction:column', 'user-select:none', '-webkit-user-select:none',
        ].join(';'))
        // 工具栏
        const tb = h('div', 'display:flex;align-items:center;gap:6px;padding:8px 10px;flex-wrap:wrap')
        const TB_BTN = BTN_CSS + ';font-size:13px;padding:3px 9px'
        const toolBtns = {}
        const mkTool = (id, label, title) => {
          const b = h('button', TB_BTN, label)
          b.title = title
          b.addEventListener('click', () => { tool = id; syncTools() })
          toolBtns[id] = b
          return b
        }
        const syncTools = () => {
          for (const id of Object.keys(toolBtns)) {
            toolBtns[id].style.borderColor = id === tool ? '#f472b6' : 'rgba(255,255,255,0.15)'
            toolBtns[id].style.color = id === tool ? '#f472b6' : 'rgba(255,255,255,0.75)'
          }
        }
        tb.append(
          mkTool('brush', '🖌 笔刷', '涂抹重绘区域'),
          mkTool('rect', '⬜ 矩形', '拖框填充重绘区域'),
          mkTool('erase', '◻ 橡皮', '擦除已涂区域'),
        )
        const sizeLab = h('span', 'font-size:10px;color:rgba(255,255,255,0.5)', '笔刷 ' + brushSize)
        const sizeRng = h('input', 'width:90px;accent-color:#f472b6')
        sizeRng.type = 'range'; sizeRng.min = 4; sizeRng.max = 200; sizeRng.step = 1; sizeRng.value = brushSize
        sizeRng.addEventListener('input', () => { brushSize = Number(sizeRng.value); sizeLab.textContent = '笔刷 ' + brushSize })
        const undoBtn = h('button', TB_BTN, '↩ 撤销')
        undoBtn.title = '撤销最后一笔'
        undoBtn.addEventListener('click', () => { wk.pop(); redraw() })
        const clearBtn = h('button', TB_BTN, '🗑 清除')
        clearBtn.title = '清空全部笔画'
        clearBtn.addEventListener('click', () => { wk.length = 0; redraw() })
        const spacer = h('span', 'flex:1')
        const saveMsg = h('span', 'font-size:10px;color:#fb7185;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap')
        const cancelBtn = h('button', TB_BTN, '取消')
        cancelBtn.addEventListener('click', () => close(false))
        const doneBtn = h('button', TB_BTN + ';color:#34d399;border-color:rgba(52,211,153,0.5)', '✔ 完成')
        doneBtn.title = '保存笔画并关闭（Esc = 取消）'
        doneBtn.addEventListener('click', () => close(true))
        tb.append(sizeLab, sizeRng, undoBtn, clearBtn, spacer, saveMsg, cancelBtn, doneBtn)

        // 画布区：底图 canvas + 笔画 canvas 双层绝对定位，view 变换同步两层
        const area = h('div', 'position:relative;flex:1;overflow:hidden;touch-action:none')
        const bcv = h('canvas', 'position:absolute;left:0;top:0')
        const pcv = h('canvas', 'position:absolute;left:0;top:0;cursor:crosshair;opacity:0.55')
        area.append(bcv, pcv)
        overlay.append(tb, area)

        const view = { scale: 1, tx: 0, ty: 0 }
        const fit = () => {
          const aw = area.clientWidth || 360, ah = area.clientHeight || 480
          view.scale = Math.min(aw / W0, ah / H0) * 0.94
          view.tx = (aw - W0 * view.scale) / 2
          view.ty = (ah - H0 * view.scale) / 2
        }
        const apply = () => {
          const w = Math.max(1, Math.round(W0 * view.scale))
          const hh = Math.max(1, Math.round(H0 * view.scale))
          for (const cv of [bcv, pcv]) {
            cv.style.left = view.tx + 'px'
            cv.style.top = view.ty + 'px'
            cv.style.width = w + 'px'
            cv.style.height = hh + 'px'
          }
          // 内部像素封顶 2048（防大图高倍缩放占内存），CSS 负责拉伸显示
          const iw = Math.min(w, 2048), ih = Math.min(hh, Math.round(2048 * H0 / W0))
          if (bcv.width !== iw || bcv.height !== ih) { bcv.width = iw; bcv.height = ih; drawBase() }
          if (pcv.width !== iw || pcv.height !== ih) { pcv.width = iw; pcv.height = ih }
        }
        const drawBase = () => {
          const ctx = bcv.getContext('2d')
          ctx.fillStyle = '#0b1220'
          ctx.fillRect(0, 0, bcv.width, bcv.height)
          if (baseImg) { ctx.drawImage(baseImg, 0, 0, bcv.width, bcv.height); return }
          // 空白参照：网格底 + 提示
          ctx.strokeStyle = 'rgba(255,255,255,0.06)'
          ctx.lineWidth = 1
          const step = Math.max(8, Math.round(64 * bcv.width / W0))
          ctx.beginPath()
          for (let x = step; x < bcv.width; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, bcv.height) }
          for (let y = step; y < bcv.height; y += step) { ctx.moveTo(0, y); ctx.lineTo(bcv.width, y) }
          ctx.stroke()
          ctx.fillStyle = 'rgba(255,255,255,0.35)'
          ctx.font = Math.round(bcv.width / 24) + 'px sans-serif'
          ctx.textAlign = 'center'
          ctx.fillText(`空白参照 ${W0}×${H0}`, bcv.width / 2, bcv.height / 2 - 8)
          ctx.font = Math.round(bcv.width / 32) + 'px sans-serif'
          ctx.fillText('在节点上填本地底图或执行一次上游节点即可衬图绘制', bcv.width / 2, bcv.height / 2 + Math.round(bcv.width / 20))
        }
        const redraw = () => {
          const k = pcv.width / W0
          const ctx = pcv.getContext('2d')
          ctx.clearRect(0, 0, pcv.width, pcv.height)
          // 全不透明绘制（半透明观感由 pcv 的 CSS opacity 提供），否则橡皮的
          // destination-out 每次只能擦掉 55%，永远擦不干净
          for (const st of wk) drawStroke(ctx, st, k)
          if (curStroke) drawStroke(ctx, curStroke, k)
        }

        // 手势：单指/鼠标 = 画，双指捏合 = 缩放平移（进行中的一笔丢弃），滚轮缩放
        const pointers = new Map()
        let pinch0 = null
        const toImg = (e) => {
          const r = area.getBoundingClientRect()
          return [
            Math.min(W0, Math.max(0, (e.clientX - r.left - view.tx) / view.scale)),
            Math.min(H0, Math.max(0, (e.clientY - r.top - view.ty) / view.scale)),
          ]
        }
        area.addEventListener('pointerdown', (e) => {
          if (e.target.closest('button, input')) return
          pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
          try { area.setPointerCapture(e.pointerId) } catch (_) {}
          if (pointers.size === 2) {
            curStroke = null // 双指手势打断进行中的一笔
            const pts = [...pointers.values()]
            pinch0 = {
              dist: Math.max(Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y), 1),
              scale: view.scale,
              mx: (pts[0].x + pts[1].x) / 2, my: (pts[0].y + pts[1].y) / 2,
              tx: view.tx, ty: view.ty,
            }
            redraw()
          } else if (pointers.size === 1) {
            const [x0, y0] = toImg(e)
            const x = Math.round(x0 * 10) / 10, y = Math.round(y0 * 10) / 10
            curStroke = tool === 'rect'
              ? { type: 'rect', x, y, w: 0, h: 0 }
              : { type: tool === 'erase' ? 'erase' : 'brush', size: brushSize, points: [[x, y]] }
            redraw()
          }
          e.preventDefault()
        })
        area.addEventListener('pointermove', (e) => {
          if (!pointers.has(e.pointerId)) return
          pointers.set(e.pointerId, { x: e.clientX, y: e.clientY })
          if (pointers.size === 2 && pinch0) {
            const pts = [...pointers.values()]
            const d = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y)
            const mx = (pts[0].x + pts[1].x) / 2, my = (pts[0].y + pts[1].y) / 2
            const r = area.getBoundingClientRect()
            // 以双指中点为锚缩放
            const ns = Math.min(12, Math.max(0.05, pinch0.scale * d / pinch0.dist))
            view.tx = mx - r.left - (pinch0.mx - r.left - pinch0.tx) * (ns / pinch0.scale)
            view.ty = my - r.top - (pinch0.my - r.top - pinch0.ty) * (ns / pinch0.scale)
            view.scale = ns
            apply(); redraw()
          } else if (curStroke) {
            const [x, y] = toImg(e)
            if (curStroke.type === 'rect') {
              curStroke.w = x - curStroke.x
              curStroke.h = y - curStroke.y
            } else {
              curStroke.points.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10])
            }
            redraw()
          }
          e.preventDefault()
        })
        const endPointer = (e) => {
          pointers.delete(e.pointerId)
          if (pointers.size < 2) pinch0 = null
          if (pointers.size === 0 && curStroke) {
            const s = curStroke
            curStroke = null
            const empty = s.type === 'rect' ? (Math.abs(s.w) < 1 && Math.abs(s.h) < 1) : !s.points.length
            if (!empty) wk.push(s)
            redraw()
          }
        }
        area.addEventListener('pointerup', endPointer)
        area.addEventListener('pointercancel', endPointer)
        area.addEventListener('wheel', (e) => {
          e.preventDefault()
          const r = area.getBoundingClientRect()
          const cx = e.clientX - r.left, cy = e.clientY - r.top
          const ns = Math.min(12, Math.max(0.05, view.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)))
          view.tx = cx - (cx - view.tx) * (ns / view.scale)
          view.ty = cy - (cy - view.ty) * (ns / view.scale)
          view.scale = ns
          apply(); redraw()
        }, { passive: false })

        const onKey = (e) => { if (e.key === 'Escape') close(false) }
        document.addEventListener('keydown', onKey)
        let closed = false
        const close = (save) => {
          if (closed) return
          if (!save) {
            closed = true
            document.removeEventListener('keydown', onKey)
            overlay.remove()
            closePaintEditor = null
            return
          }
          // 保存：先抽稀（实测 5.5x 减体积），再落盘/写回
          const payload = JSON.stringify({ strokes: simplifyStrokes(wk), image_size: [W0, H0] })
          doneBtn.disabled = true
          doneBtn.textContent = '⏳ 保存…'
          saveMsg.textContent = ''
          saveStrokes(payload, (err, path) => {
            if (err) {
              // 失败不关弹窗、不写参数：宁可让用户重试，也不能丢掉已画内容
              doneBtn.disabled = false
              doneBtn.textContent = '✔ 完成'
              saveMsg.textContent = '❌ 保存失败：' + err
              return
            }
            closed = true
            document.removeEventListener('keydown', onKey)
            overlay.remove()
            closePaintEditor = null
            if (REF_KEY) {
              // 原值是绑定（{{ Param.x }}）⇒ 本次保存把节点参数改成直填路径，提示行明说
              rebindDone = isWired(String((cur.params || {})[REF_KEY] || ''))
              setParams({ [P.strokesKey]: '', [REF_KEY]: path })
            }
            // 重新从参数解析（外置：按新路径重读，保证缩略图 = 磁盘内容）
            strokesSrc = null
            syncStrokes()
            thumbKey = ''
            renderThumb()
            updateHint()
          })
        }
        closePaintEditor = () => close(false)

        syncTools()
        document.body.appendChild(overlay)
        fit(); apply(); redraw()
      }
      editBtn.addEventListener('click', openEditor)
      thumb.addEventListener('click', () => { if (editable()) openEditor() })

      refreshers.push(() => {
        syncStrokes()
        // 底图三条链：① 回放/执行态：resolved inputs + 节点级代理（同 compareInput）；
        // ② 编辑态：paramSources 运行时值解析出对象 id + 通用代理 blob 拉取
        // （同模型下拉链路），执行过一次上游即可衬图绘制；
        // ③ 本地路径兜底（P.imagePathKey，如本节点自带的 image_path 参数）：直读
        // Studio 媒体端点（目录白名单内），无需先执行上游即可衬图绘制
        const resolved = resolvedInputs()
        const rr = resolved && resolved[P.imageInput]
        let inId = rr && (typeof rr === 'string' ? rr : (rr.$id || rr.id || ''))
        let viaProxy = ''
        if (!(inId && cur.execution)) {
          const rv = resolveParam(P.imageInput)
          const base = resolveParam('service_url')
          if (rv && !isWired(rv) && /^[0-9a-f]{32}$/.test(rv) && base && !isWired(base)) {
            inId = rv
            viaProxy = base + '|' + (resolveParam('service_token') || '')
          } else {
            inId = ''
          }
        }
        if (inId && inId !== baseFor) {
          baseFor = inId
          const img = new Image()
          img.onload = () => { baseImg = img; thumbKey = ''; renderThumb(); updateHint() }
          img.onerror = () => { if (baseFor === inId) { baseFor = ''; baseImg = null } }
          if (viaProxy) {
            const [base, tok] = viaProxy.split('|')
            fetch('/api/v1/service-proxy', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ service_url: base, service_token: tok, method: 'GET', path: '/images/' + inId, timeout_sec: 30 }),
            }).then((r) => {
              if (!r.ok) throw new Error('HTTP ' + r.status)
              return r.blob()
            }).then((b) => { img.src = URL.createObjectURL(b) })
              .catch(() => { if (baseFor === inId) { baseFor = ''; baseImg = null } })
          } else {
            img.src = nodeProxyUrl('/images/' + inId, 'GET')
          }
        } else if (!inId && P.imagePathKey) {
          // ③ 本地路径兜底：无上游对象 id（尚未执行）时读节点自己的本地图参数。
          //    必须走 resolveParam：模板节点普遍写 image_path: "{{ Param.image_path }}"，
          //    直读 cur.params 拿到模板原文 ⇒ isWired 判定为「绑上游」丢弃 ⇒ 编辑器
          //    里没有底图（wf181 事故）。resolveParam 对绑流水线参数返回其当前值，
          //    与 kind:'file' 控件的预览（effPath）同一个口径。
          const lp = paramText(P.imagePathKey)
          const fileKey = lp && !isWired(lp) ? 'file:' + lp : ''
          if (fileKey && fileKey !== baseFor) {
            baseFor = fileKey
            const img = new Image()
            img.onload = () => { baseImg = img; thumbKey = ''; renderThumb(); updateHint() }
            img.onerror = () => { if (baseFor === fileKey) { baseFor = ''; baseImg = null } }
            img.src = mediaUrl(lp)
          } else if (!fileKey && baseFor.indexOf('file:') === 0) {
            baseFor = ''; baseImg = null
          }
        }
        const key = strokesSrc + '|' + (strokesLoading ? 'L' : '') + '|' + strokesErr + '|'
          + strokes.length + '|' + baseFor + '|' + (baseImg ? 1 : 0)
        if (key !== thumbKey) { thumbKey = key; renderThumb() }
        updateHint()
      })
    }

    // ---- 点/框选画布（spec.pointSelect）：SAM 交互式分割提示 ----
    // 点=正样本、Shift/Alt/右键点=负样本、拖拽=框选；坐标以**原图坐标**写回
    // points/box 参数（服务端无需换算）。底图走 compareInput 同款链路
    // （outputs.__inputs_resolved + 节点级代理），结果蒙版有预览帧时叠加显示。
    const psWrap = h('div', 'display:none;margin-bottom:6px')
    if (spec.pointSelect) {
      const P = spec.pointSelect
      const psBox = h('div', 'position:relative;width:100%;border-radius:8px;overflow:hidden;background:rgba(0,0,0,0.35);user-select:none;touch-action:none')
      const psBase = h('img', 'width:100%;display:block;pointer-events:none')
      const psMask = h('img', 'position:absolute;inset:0;width:100%;height:100%;object-fit:fill;opacity:0.45;display:none;pointer-events:none')
      const psCv = h('canvas', 'position:absolute;inset:0;width:100%;height:100%;touch-action:none')
      const psTag = h('span', 'position:absolute;left:4px;top:4px;font-size:9px;background:rgba(0,0,0,0.6);border-radius:4px;padding:1px 5px;color:rgba(255,255,255,0.85);pointer-events:none', '点=正 / Shift点=负 / 拖拽=框选')
      psBox.append(psBase, psMask, psCv, psTag)
      const psHint = h('div', 'font-size:9px;color:rgba(255,255,255,0.4);margin-top:2px')
      const psRow = h('div', 'display:flex;gap:6px;align-items:center;margin-top:4px')
      const psClear = h('button', BTN_CSS, '清空提示')
      const psMaskBtn = h('button', BTN_CSS, '显示结果蒙版')
      psRow.append(psClear, psMaskBtn)
      psWrap.append(psBox, psRow, psHint)

      let psW = 0, psH = 0
      let psPts = [], psBoxSel = null, psDragging = false, psStart = null, psMoved = false
      const psParse = () => {
        psPts = []
        for (const chunk of String(readParam(P.pointsKey, '') || '').split(';')) {
          const p = chunk.split(',').map((x) => x.trim()).filter((x) => x !== '')
          if (p.length >= 2 && !isNaN(Number(p[0])) && !isNaN(Number(p[1]))) {
            psPts.push([Number(p[0]), Number(p[1]), p.length > 2 ? Number(p[2]) : 1])
          }
        }
        const b = String(readParam(P.boxKey, '') || '').split(',').map((x) => Number(x.trim())).filter((x) => !isNaN(x))
        psBoxSel = b.length === 4 ? b : null
      }
      const psWrite = () => {
        setParam(P.pointsKey, psPts.map((p) => `${Math.round(p[0])},${Math.round(p[1])},${p[2] ? 1 : 0}`).join(';'))
        if (P.boxKey) setParam(P.boxKey, psBoxSel ? psBoxSel.map((v) => Math.round(v)).join(',') : '')
      }
      const psDraw = () => {
        if (!psW || !psH) return
        psCv.width = psW
        psCv.height = psH
        const c = psCv.getContext('2d')
        c.clearRect(0, 0, psW, psH)
        const r = Math.max(4, Math.round(Math.min(psW, psH) * 0.012))
        if (psBoxSel) {
          c.strokeStyle = '#ff4040'
          c.lineWidth = Math.max(2, r / 2)
          c.strokeRect(psBoxSel[0], psBoxSel[1], psBoxSel[2] - psBoxSel[0], psBoxSel[3] - psBoxSel[1])
        }
        for (const p of psPts) {
          c.beginPath()
          c.arc(p[0], p[1], r, 0, Math.PI * 2)
          c.fillStyle = p[2] ? '#34d399' : '#ff4040'
          c.fill()
          c.lineWidth = Math.max(1, r / 4)
          c.strokeStyle = '#000'
          c.stroke()
        }
        const pos = psPts.filter((p) => p[2]).length
        psHint.textContent = `正样本 ${pos} 点 / 负样本 ${psPts.length - pos} 点 / 框选 ${psBoxSel ? 1 : 0} 处`
          + (editable() ? '；编辑结果随下次执行生效' : '（只读）')
        psClear.disabled = !editable()
        psClear.style.opacity = editable() ? '1' : '0.5'
      }
      const psToOrig = (e) => {
        const rect = psCv.getBoundingClientRect()
        const x = (e.clientX - rect.left) / Math.max(1, rect.width) * psW
        const y = (e.clientY - rect.top) / Math.max(1, rect.height) * psH
        return [Math.max(0, Math.min(psW - 1, x)), Math.max(0, Math.min(psH - 1, y))]
      }
      psCv.addEventListener('pointerdown', (e) => {
        if (!editable() || !psW) return
        psDragging = true
        psMoved = false
        psStart = psToOrig(e)
        if (psCv.setPointerCapture) { try { psCv.setPointerCapture(e.pointerId) } catch (err) { /* 忽略 */ } }
      })
      psCv.addEventListener('pointermove', (e) => {
        if (!psDragging) return
        const p = psToOrig(e)
        if (Math.abs(p[0] - psStart[0]) + Math.abs(p[1] - psStart[1]) > Math.min(psW, psH) * 0.01) psMoved = true
        if (psMoved) {
          psBoxSel = [Math.min(psStart[0], p[0]), Math.min(psStart[1], p[1]),
                      Math.max(psStart[0], p[0]), Math.max(psStart[1], p[1])]
          psDraw()
        }
      })
      psCv.addEventListener('pointerup', (e) => {
        if (!psDragging) return
        psDragging = false
        if (psMoved) {
          psWrite()
        } else {
          const p = psToOrig(e)
          const neg = e.shiftKey || e.altKey || e.button === 2
          psPts.push([p[0], p[1], neg ? 0 : 1])
          psWrite()
        }
        psDraw()
      })
      psCv.addEventListener('contextmenu', (e) => e.preventDefault())
      psClear.addEventListener('click', () => { psPts = []; psBoxSel = null; psWrite(); psDraw() })
      let psMaskOn = false
      psMaskBtn.addEventListener('click', () => {
        psMaskOn = !psMaskOn
        psMask.style.display = psMaskOn ? 'block' : 'none'
        psMaskBtn.textContent = psMaskOn ? '隐藏结果蒙版' : '显示结果蒙版'
      })
      let psLoadedFor = ''
      refreshers.push(() => {
        const resolved = resolvedInputs()
        const ref = resolved && resolved[P.baseImage]
        const inId = ref && (typeof ref === 'string' ? ref : (ref.$id || ref.id || ''))
        if (!inId || !cur.execution) { psWrap.style.display = 'none'; psLoadedFor = ''; return }
        psWrap.style.display = 'block'
        if (psLoadedFor !== inId) {
          psLoadedFor = inId
          psBase.onload = () => { psW = psBase.naturalWidth; psH = psBase.naturalHeight; psParse(); psDraw() }
          psBase.onerror = () => { psHint.textContent = '底图加载失败（对象已驱逐或未执行过）' }
          psBase.src = nodeProxyUrl('/images/' + inId, 'GET')
        }
        psParse()
        psDraw()
        if (psMaskOn) {
          const pv = cur.preview
          const url = pv && pv.url
          if (url) psMask.src = url
        }
      })
    }

    // ---- 蒙版网格点选（spec.maskMontage）：自动分割结果挑选保留/排除 ----
    // 服务端把各蒙版拼成编号网格图（montage_tile/montage_cols 由算子回传，
    // 保证格子坐标与后端一致），点格子 → 切换 keep 下标 → setParam 写回。
    const mmWrap = h('div', 'display:none;margin-bottom:6px')
    if (spec.maskMontage) {
      const P = spec.maskMontage
      const mmGrid = h('img', 'width:100%;display:block;border-radius:6px;cursor:crosshair;background:rgba(0,0,0,0.3)')
      const mmChips = h('div', 'display:flex;flex-wrap:wrap;gap:4px;margin-top:4px')
      const mmHint = h('div', 'font-size:9px;color:rgba(255,255,255,0.4);margin-top:2px')
      const mmSel = h('img', 'width:100%;display:block;border-radius:6px;margin-top:4px;background:rgba(0,0,0,0.3)')
      mmWrap.append(mmGrid, mmChips, mmHint, mmSel)
      let mmN = 0, mmCols = 1, mmTile = 160, mmLoadedFor = '', mmSelFor = ''
      const mmKeep = () => {
        const s = new Set()
        for (const x of String(readParam(P.keepKey, '') || '').split(',')) {
          const v = x.trim()
          if (v !== '' && !isNaN(Number(v))) s.add(Number(v))
        }
        return s
      }
      const mmChipsRender = () => {
        mmChips.textContent = ''
        const ks = mmKeep()
        mmHint.textContent = ks.size === 0
          ? `${mmN} 个蒙版（未选择 = 全选）`
          : `已选 ${ks.size}/${mmN}：${[...ks].sort((a, b) => a - b).join(', ')}（随下次执行生效）`
        for (let i = 0; i < Math.min(mmN, 64); i++) {
          const on = ks.has(i)
          const chip = h('span', `font-size:9px;padding:1px 5px;border-radius:4px;cursor:pointer;background:${on ? 'rgba(52,211,153,0.85)' : 'rgba(255,255,255,0.12)'};color:${on ? '#06251a' : 'rgba(255,255,255,0.7)'}`, String(i))
          chip.addEventListener('click', () => {
            const s = mmKeep()
            if (s.has(i)) s.delete(i)
            else s.add(i)
            setParam(P.keepKey, [...s].sort((a, b) => a - b).join(','))
            mmChipsRender()
          })
          mmChips.append(chip)
        }
      }
      mmGrid.addEventListener('click', (e) => {
        if (!editable() || !mmN) return
        const rect = mmGrid.getBoundingClientRect()
        const gx = (e.clientX - rect.left) / Math.max(1, rect.width) * mmGrid.naturalWidth
        const gy = (e.clientY - rect.top) / Math.max(1, rect.height) * mmGrid.naturalHeight
        const col = Math.floor(gx / mmTile)
        const idx = Math.floor(gy / mmTile) * mmCols + col
        if (col < 0 || col >= mmCols || idx < 0 || idx >= mmN) return
        const s = mmKeep()
        if (s.has(idx)) s.delete(idx)
        else s.add(idx)
        setParam(P.keepKey, [...s].sort((a, b) => a - b).join(','))
        mmChipsRender()
      })
      refreshers.push(() => {
        const o = cur.outputs || {}
        mmN = Number(o.count || 0)
        mmCols = Math.max(1, Number(o.montage_cols || 1))
        mmTile = Math.max(1, Number(o.montage_tile || 160))
        const mid = o[P.montageKey]
        const sid = o[P.selectedKey]
        if (!mid || !cur.execution || !mmN) { mmWrap.style.display = 'none'; mmLoadedFor = ''; mmSelFor = ''; return }
        mmWrap.style.display = 'block'
        if (mmLoadedFor !== mid) { mmLoadedFor = mid; mmGrid.src = nodeProxyUrl('/images/' + mid, 'GET') }
        if (sid && mmSelFor !== sid) { mmSelFor = sid; mmSel.src = nodeProxyUrl('/images/' + sid, 'GET') }
        mmChipsRender()
      })
    }

    // ---- 检测框列表（spec.boxList）：标签 + 置信度 + 框尺寸 ----
    const blWrap = h('div', 'display:none;margin-bottom:6px')
    if (spec.boxList) {
      const P = spec.boxList
      const blList = h('div', 'font-size:10px;color:rgba(255,255,255,0.8);max-height:160px;overflow:auto')
      blWrap.append(blList)
      // 值可能带 '@' 负载信封（防 Studio 自动反序列化），解析前先剥掉
      const blJ = (v) => {
        const t = String(v == null ? '' : v)
        try { return JSON.parse(t.startsWith('@') ? t.slice(1) : t) } catch (err) { return null }
      }
      refreshers.push(() => {
        const o = cur.outputs || {}
        const labs = blJ(o[P.labelsKey]) || []
        const scs = blJ(o[P.scoresKey]) || []
        const bxs = blJ(o[P.boxesKey]) || []
        if (!bxs.length) { blWrap.style.display = 'none'; return }
        blWrap.style.display = 'block'
        blList.textContent = ''
        blList.append(h('div', 'font-size:9px;color:rgba(255,255,255,0.45)', `命中 ${bxs.length} 个（门限内，按置信度降序）`))
        bxs.forEach((b, i) => {
          const sc = scs[i] !== undefined ? Number(scs[i]).toFixed(3) : '-'
          const w = Math.round(b[2] - b[0])
          const hh = Math.round(b[3] - b[1])
          blList.append(h('div', 'padding:2px 0;border-bottom:1px solid rgba(255,255,255,0.06)',
            `#${i + 1} ${labs[i] || '?'}  ${sc}  [${Math.round(b[0])},${Math.round(b[1])} ${w}×${hh}]`))
        })
      })
    }

    // ---- 切片网格读数（spec.tileGridOverlay）：网格图 + 块数/行列/预估 ----
    const tgWrap = h('div', 'display:none;margin-bottom:6px')
    if (spec.tileGridOverlay) {
      const P = spec.tileGridOverlay
      const tgImg = h('img', 'width:100%;display:block;border-radius:6px;background:rgba(0,0,0,0.3)')
      const tgInfo = h('div', 'font-size:10px;color:rgba(255,255,255,0.75);margin-top:2px')
      tgWrap.append(tgImg, tgInfo)
      let tgLoadedFor = ''
      refreshers.push(() => {
        const o = cur.outputs || {}
        const gid = o[P.gridKey]
        if (!gid || !cur.execution) { tgWrap.style.display = 'none'; tgLoadedFor = ''; return }
        tgWrap.style.display = 'block'
        if (tgLoadedFor !== gid) { tgLoadedFor = gid; tgImg.src = nodeProxyUrl('/images/' + gid, 'GET') }
        const parts = []
        if (o.tiles) parts.push(`块数 ${o.tiles}`)
        if (o.rows) parts.push(`${o.rows}行×${o.cols}列`)
        if (o.target_size) parts.push(`目标 ${o.target_size}`)
        if (o.est_samples) parts.push(`预估采样步 ${o.est_samples}`)
        tgInfo.textContent = parts.join('　')
      })
    }

    // ---- ▶预览（op-replay 重放）：已移除 ----
    // 重放依赖推理服务内存中的上次输入对象，对象 TTL（默认 1h）到期后必然失败，对
    // 「看历史结果」无意义，故移除入口；历史结果一律走物化预览帧（媒体缓存，TTL 72h）。
    const replayRow = h('div', 'display:none;align-items:center;margin-bottom:6px')

    // ---- 实时预览：默认开启，不提供开关 ----
    // preview_every 由执行器默认 1（>0 即开）兜底；过程帧是物化历史预览的数据来源，
    // 故不再提供开关/参数控件（避免被关掉导致无历史预览帧）。
    const peRow = h('div', 'margin-bottom:6px;display:none')

    // ---- 参数控件区（advanced:true 的字段收进折叠区，保持节点紧凑）----
    const controls = h('div', 'margin-bottom:6px')
    const advBox = h('div', 'display:none;margin:2px 0 6px;padding:5px 6px;border:1px solid rgba(255,255,255,0.08);border-radius:6px')
    let advCount = 0
    for (const f of spec.fields || []) {
      const c = controlOf(f)
      if (f.advanced) { advBox.append(c); advCount++ } else controls.append(c)
    }
    if (advCount) {
      const advToggle = h('button', BTN_CSS + ';width:100%;text-align:left;margin:2px 0 4px', `⚙ 连接/高级（${advCount}）▸`)
      let advOpen = false
      advToggle.addEventListener('click', () => {
        advOpen = !advOpen
        advBox.style.display = advOpen ? 'block' : 'none'
        advToggle.textContent = `⚙ 连接/高级（${advCount}）${advOpen ? '▾' : '▸'}`
      })
      controls.append(advToggle, advBox)
    }

    // ---- 本地媒体显示区（可选 spec.mediaImageParam / mediaImageOutput / mediaVideoOutput）----
    // 直读 FlowX 所在机器的本地文件（Studio 同源媒体端点），图片点击进可缩放
    // lightbox，视频内嵌播放 + ⛶ 放大。与 preview 帧互补：preview 是执行过程
    // 帧（推理服务内存缓冲，易失），本区是落盘结果/参数源文件（持久）。
    const mediaBox = h('div', 'display:none;position:relative;margin-bottom:4px;border-radius:8px;overflow:hidden;background:rgba(0,0,0,0.3)')
    const mediaImg = h('img', 'display:none;width:100%;border-radius:8px;cursor:zoom-in')
    const mediaVideo = h('video', 'display:none;width:100%;background:#000;border-radius:8px')
    mediaVideo.controls = true
    mediaVideo.loop = true
    mediaVideo.muted = true
    mediaVideo.playsInline = true
    const mediaExpand = h('button', BTN_CSS + ';position:absolute;top:6px;right:6px;display:none', '⛶')
    mediaExpand.title = '放大播放'
    const mediaPlaceholder = h('div', 'padding:14px 8px;text-align:center;font-size:10px;color:rgba(255,255,255,0.3)')
    mediaBox.append(mediaImg, mediaVideo, mediaPlaceholder, mediaExpand)
    const mediaPath = h('div', 'color:rgba(255,255,255,0.4);margin-bottom:6px;word-break:break-all;font-size:10px')

    if (spec.mediaImageParam || spec.mediaImageOutput || spec.mediaVideoOutput) {
      mediaImg.addEventListener('click', () => {
        if (!mediaImg.src || mediaImg.style.display === 'none') return
        setLightbox(buildImageLightbox(mediaImg.src, mediaPath.textContent, closeLightbox))
      })
      mediaExpand.addEventListener('click', () => {
        if (!mediaVideo.src) return
        const t = mediaVideo.currentTime
        const lb = buildVideoLightbox(mediaVideo.src, mediaPath.textContent, t, () => {
          try { mediaVideo.currentTime = lb.video.currentTime } catch (_) {}
          closeLightbox()
        })
        setLightbox(lb.overlay)
      })
      mediaImg.addEventListener('error', () => {
        mediaImg.style.display = 'none'
        mediaPlaceholder.style.display = 'block'
        mediaPlaceholder.textContent = '文件不可读（路径失效或不在媒体白名单）'
      })
    }

    // ---- 输出区 ----
    const outBox = h('div', 'color:rgba(255,255,255,0.45);word-break:break-all')

    // 布局顺序：状态行 → 图像相关区（过程/结果预览帧、前后对比、本地图片/视频、
    // ▶重放、实时预览开关）→ 参数控件 → 输出。图像预览一律置顶，手机窄画布上
    // 不必往下滚就能看到结果；无预览/无媒体/无重放的节点这些块是 display:none，
    // 不占位也不改变观感。
    el.append(header, prevWrap, cmpWrap, paintWrap, psWrap, mmWrap, blWrap, tgWrap, mediaBox, mediaPath, replayRow, peRow, controls, outBox)
    if (spec.note) el.append(h('div', 'font-size:9px;color:rgba(255,255,255,0.3);margin-top:4px', spec.note))

    let lastPreviewUrl = ''
    function render(p) {
      cur = p
      dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
      statusLabel.textContent = p.status
      const pv = p.preview
      if (pv && pv.url) {
        prevWrap.style.display = 'block'
        const effective = pv.url
        if (effective !== lastPreviewUrl) {
          lastPreviewUrl = effective
          prevImg.src = effective
        }
        const prog = typeof pv.progress === 'number' ? pv.progress : null
        const running = p.status === 'running' && (prog === null || prog < 1)
        progOuter.style.display = running ? 'block' : 'none'
        stopBtn.style.display = running ? 'inline-block' : 'none'
        if (prog !== null) progInner.style.width = Math.round(prog * 100) + '%'
        if (!prevInfo.textContent || running) {
          prevInfo.textContent = prog !== null ? `进度 ${Math.round(prog * 100)}%` : ''
        }
      } else {
        // 无预览帧：不展示（媒体一律经预览帧中转 GET preview-frame 或媒体代理
        // service-proxy 拉取；outputs/metadata 禁止内嵌 base64/data URL）
        prevWrap.style.display = 'none'
        lastPreviewUrl = ''
      }
      const o = p.outputs || {}
      // 本地媒体显示：参数源文件（load-image 类）或落盘结果（save-* 类）
      if (spec.mediaImageParam || spec.mediaImageOutput || spec.mediaVideoOutput) {
        const isVideo = !!spec.mediaVideoOutput
        const fromParam = !!spec.mediaImageParam
        const path = fromParam
          ? String((p.params || {})[spec.mediaImageParam] || '')
          : String(o[spec.mediaImageOutput || spec.mediaVideoOutput] || '')
        const wired = fromParam && isWired(path)
        if (path && !wired) {
          const src = mediaUrl(path)
          mediaBox.style.display = 'block'
          mediaPlaceholder.style.display = 'none'
          mediaImg.style.display = isVideo ? 'none' : 'block'
          mediaVideo.style.display = isVideo ? 'block' : 'none'
          mediaExpand.style.display = isVideo ? 'inline-block' : 'none'
          const el2 = isVideo ? mediaVideo : mediaImg
          if (el2.dataset.path !== path) { el2.src = src; el2.dataset.path = path }
          mediaPath.textContent = path + (o.size_bytes ? `（${(Number(o.size_bytes) / 1024).toFixed(0)}KB）` : '')
        } else if (fromParam) {
          // 无路径或绑定态：占位提示
          mediaBox.style.display = 'block'
          mediaImg.style.display = 'none'
          mediaVideo.style.display = 'none'
          mediaExpand.style.display = 'none'
          mediaPlaceholder.style.display = 'block'
          mediaPlaceholder.textContent = wired ? '图片路径由绑定提供，执行后可见' : '未设置图片路径'
          mediaPath.textContent = ''
        } else {
          mediaBox.style.display = 'none'
          mediaPath.textContent = ''
        }
      }
      outBox.textContent = ''
      const entries = Object.entries(o).filter(([k]) => !k.startsWith('__'))
      if (entries.length === 0) {
        outBox.textContent = '等待执行…'
      } else {
        for (const [k, v] of entries) {
          const shown = String(v)
          outBox.append(h('div', '', `${k}: ${(shown.startsWith('@') ? shown.slice(1) : shown).slice(0, 80)}`))
        }
      }
      refreshers.forEach((f) => f())
    }

    render(props)

    return {
      update(next) { render(next) },
      unmount() { if (closePaintEditor) closePaintEditor(); closeLightbox(); closeTextEditor(); document.removeEventListener('keydown', onEsc); el.textContent = '' },
    }
  }
}

// 测试出口（_tools/test-widget-ui.mjs 直接验抽稀效果；发布产物为单文件 bundle，
// 多挂一个静态属性不影响运行时行为）
createNodeWidget.simplifyStrokes = simplifyStrokes
createNodeWidget.simplifyPoints = simplifyPoints

// ---- 节点专属定义（widget-def.js，由 build-widget.py 拼接，请勿直接编辑本文件）----
const WIDGET_SPEC = {
  icon: '🌑',
  title: '环境投影（落地阴影）',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true, advanced: true },
    { key: 'image', label: '主体图像对象 ID（image）', kind: 'text', mono: true },
    { key: 'mask', label: '主体蒙版对象 ID（留空用 alpha）', kind: 'text', mono: true, advanced: true },
    { key: 'angle', label: '阴影方向（度，90=正下方）', kind: 'slider', min: 0, max: 360, step: 5, default: 90 },
    { key: 'distance', label: '阴影距离(px)', kind: 'slider', min: 0, max: 300, step: 2, default: 24 },
    { key: 'blur', label: '模糊半径(px)', kind: 'slider', min: 0, max: 120, step: 2, default: 24 },
    { key: 'opacity', label: '不透明度', kind: 'slider', min: 0.0, max: 1.0, step: 0.05, default: 0.45 },
    { key: 'squash', label: '压扁（贴地透视，1=不压）', kind: 'slider', min: 0.1, max: 1.0, step: 0.05, default: 0.55 },
    { key: 'fade', label: '远端淡出', kind: 'slider', min: 0.0, max: 1.0, step: 0.05, default: 0.0 },
    { key: 'color', label: '投影颜色', kind: 'text', mono: true, default: '#000000' },
    { key: 'canvas', label: '画布模式', kind: 'select', options: ['expand', 'keep'], default: 'expand' },
    { key: 'background', label: '底图对象 ID（留空输出透明画布）', kind: 'text', mono: true, advanced: true },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true, advanced: true },
  ],
  replay: ['angle', 'distance', 'blur', 'opacity', 'squash', 'fade', 'color', 'canvas'],
  compareInput: 'image',
  note: 'shadow_box 之外逐字节不变（可复核）；canvas=expand 主体像素逐字节保留。'
        + '商品图有 alpha 时不必接蒙版节点。',
};

export default createNodeWidget(WIDGET_SPEC)
