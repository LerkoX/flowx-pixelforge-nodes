#!/usr/bin/env node
/**
 * test-widget-ui.mjs：画布侧节点 widget（ui/node-widget.js）交互回归。
 *
 * 覆盖的是「用 DOM stub + 假 XHR 在 node 里真跑一遍 bundle」的可测部分——
 * 按钮是否渲染、上传是否写回参数、弹层能不能关。这些都不是 python 门槛
 * （check-bundle.py）能看见的：产物字节完全合法、语法也没问题，但用户点下去
 * 关不掉弹窗。
 *
 * 已入网的回归点（每条都对应一次真实事故）：
 *   1. kind:'file' 上传控件：选文件 → POST /api/v1/media/upload → 返回路径写回参数
 *   2. 绑定态（{{ }}）/ 回放态（无 onParamsChange）不得出现可写上传入口
 *   3. 未声明 kind:'file' 的处理类节点不得出现任何上传入口（能力共享、展示声明制）
 *   4. 预览图放大弹层的三条关闭途径（✕ / 点遮罩空白 / Esc）——1.3.0 事故：
 *      fileControl 调 buildImageLightbox 漏传 onClose ⇒ 弹窗关不掉；且 Esc 监听
 *      原先挂在 mediaImageParam 分支里，load-image@1.3.0 正好去掉该声明 ⇒ Esc 也失效
 *   5. 落盘结果媒体区（mediaImageOutput/mediaVideoOutput）的弹层同源同修
 *
 * 节点清单是**扫出来的**（谁声明 kind:'file' / media*Output 就测谁），
 * 新增加载类节点自动纳入，不用改本文件。
 *
 * 用法：node nodes/_tools/test-widget-ui.mjs [节点目录名...]   （退出码 0=通过）
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const NODES = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const only = process.argv.slice(2)

// ---------- DOM / 浏览器 API stub ----------
// 够 widget-base 跑通渲染与事件：cssText 要拆成逐条声明（遮罩靠 style.position
// 判定）、append/remove 要真维护父子关系（弹层靠 overlay.remove() 关）。
const ctx2d = new Proxy({}, { get: (t, k) => (k === 'canvas' ? {} : () => {}) })
function mkEl(tag) {
  return {
    tagName: tag, children: [], dataset: {}, _text: '', _h: {},
    style: new Proxy({}, {
      set: (t, k, v) => {
        if (k === 'cssText') {
          for (const d of String(v).split(';')) {
            const i = d.indexOf(':')
            if (i > 0) t[d.slice(0, i).trim()] = d.slice(i + 1).trim()
          }
        }
        t[k] = v
        return true
      },
      get: (t, k) => t[k] || '',
    }),
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    append(...n) { n.forEach((c) => { if (c && typeof c === 'object') c.parent = this }); this.children.push(...n) },
    appendChild(n) { if (n) n.parent = this; this.children.push(n); return n },
    insertBefore(n) { if (n) n.parent = this; this.children.push(n); return n },
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1) },
    remove() { if (this.parent) this.parent.removeChild(this) },
    addEventListener(ev, fn) { (this._h[ev] = this._h[ev] || []).push(fn) },
    removeEventListener(ev, fn) {
      const l = this._h[ev] || []; const i = l.indexOf(fn); if (i >= 0) l.splice(i, 1)
    },
    setAttribute() {}, getAttribute: () => null, removeAttribute() {},
    querySelector: () => null, querySelectorAll: () => [],
    getContext: () => ctx2d, getBoundingClientRect: () => ({ width: 10, height: 10, left: 0, top: 0 }),
    focus() {}, blur() {}, click() {}, setPointerCapture() {}, releasePointerCapture() {},
    play: () => Promise.resolve(), pause() {}, requestFullscreen: () => Promise.resolve(),
    get textContent() { return this._text }, set textContent(v) { this._text = v; this.children = [] },
    get innerHTML() { return '' }, set innerHTML(v) {},
  }
}
const docHandlers = {}
global.document = {
  createElement: mkEl, createTextNode: (t) => ({ textContent: t }),
  body: mkEl('body'), activeElement: null,
  addEventListener(ev, fn) { (docHandlers[ev] = docHandlers[ev] || []).push(fn) },
  removeEventListener(ev, fn) {
    const l = docHandlers[ev] || []; const i = l.indexOf(fn); if (i >= 0) l.splice(i, 1)
  },
  querySelector: () => null,
}
global.window = { addEventListener() {}, removeEventListener() {}, devicePixelRatio: 1 }
global.Image = class { set src(v) {} }
global.URL = { createObjectURL: () => 'blob:x' }
global.fetch = async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => '' })
let lastSentForm = null
let lastUploadUrl = ''
global.XMLHttpRequest = class {
  constructor() { this.upload = {}; this.status = 0 }
  open(_m, url) { this.url = url }
  setRequestHeader() {}
  send(fd) {
    lastSentForm = fd
    lastUploadUrl = String(this.url || '')
    // 按落点返回不同文件名：paint 笔画走 subdir=strokes，加载类节点上传图片
    const strokes = lastUploadUrl.includes('subdir=strokes')
    this.status = 200
    this.responseText = JSON.stringify({ code: 200, data: strokes
      ? { path: '~/flowx-input/uploads/strokes/T_x.json', name: 'T_x.json' }
      : { path: '~/flowx-input/uploads/T_x.png', name: 'T_x.png' } })
    this.onload && this.onload()
  }
}
global.FormData = class { constructor() { this.entries = [] } append(k, v) { this.entries.push([k, v]) } }

// ---------- 断言小工具 ----------
let fails = 0
const check = (label, cond) => { console.log((cond ? 'PASS ' : 'FAIL ') + label); if (!cond) fails++ }
const stubEvent = () => ({ stopPropagation() {}, preventDefault() {}, target: { closest: () => null } })
const fire = (el, ev, arg) => { if (el) (el._h[ev] || []).forEach((fn) => fn(arg === undefined ? stubEvent() : arg)) }
const esc = () => (docHandlers.keydown || []).forEach((fn) => fn({ key: 'Escape' }))
const overlays = () => document.body.children.filter((c) => String(c.style.position) === 'fixed')
function findAllByTag(el, tag, out = []) {
  if (el && el.tagName === tag) out.push(el)
  for (const c of (el && el.children) || []) findAllByTag(c, tag, out)
  return out
}
function findByText(el, txt) {
  if (!el) return null
  if (el._text === txt) return el
  for (const c of el.children || []) { const r = findByText(c, txt); if (r) return r }
  return null
}

// ---------- 载入 bundle（去掉 export 语句后当普通脚本跑）----------
function bundle(node) {
  const file = path.join(NODES, node, 'ui', 'node-widget.js')
  const src = fs.readFileSync(file, 'utf8').replace(/^export default .*$/m, '')
  return { ...new Function(src + '\nreturn { createNodeWidget, WIDGET_SPEC };')(), file }
}
function mount(node, props) {
  const { createNodeWidget, WIDGET_SPEC } = bundle(node)
  const el = mkEl('div')
  createNodeWidget(WIDGET_SPEC)(el, {
    nodeId: 'T1', nodeRef: node, status: 'idle', inputs: [], outputs: {},
    params: {}, paramSources: {}, execution: null, preview: null,
    theme: 'dark', locale: 'zh', ...props,
  })
  return el
}

// ---------- 扫节点：谁声明什么，就测什么 ----------
const defs = {}
for (const d of fs.readdirSync(NODES)) {
  const wd = path.join(NODES, d, 'ui', 'widget-def.js')
  if (fs.existsSync(wd)) defs[d] = fs.readFileSync(wd, 'utf8')
}
// 字段块 = 一对不含嵌套对象的花括号；在其中找声明 kind:'file' 的那个块
const fileNodes = Object.keys(defs).filter((d) => /kind:\s*'file'/.test(defs[d]))
const fileKey = (d) => {
  for (const block of defs[d].match(/\{[^{}]*\}/g) || []) {
    if (/kind:\s*'file'/.test(block)) return (block.match(/key:\s*'([^']+)'/) || [])[1]
  }
  return undefined
}
const mediaNodes = Object.keys(defs)
  .map((d) => {
    const img = (defs[d].match(/mediaImageOutput:\s*'([^']+)'/) || [])[1]
    const vid = (defs[d].match(/mediaVideoOutput:\s*'([^']+)'/) || [])[1]
    return img ? [d, img, false] : vid ? [d, vid, true] : null
  })
  .filter(Boolean)
const plainNode = Object.keys(defs).find((d) => !/kind:\s*'file'/.test(defs[d]))

const all = [...fileNodes, ...mediaNodes.map(([d]) => d), plainNode].filter(Boolean)
const targets = only.length ? all.filter((d) => only.includes(d)) : all
console.log(`节点：file=${fileNodes.join(',') || '-'} media=${mediaNodes.map((m) => m[0]).join(',') || '-'}`
  + ` plain=${plainNode || '-'}\n`)

// ---------- 1/2. kind:'file' 上传控件 ----------
for (const node of fileNodes.filter((d) => targets.includes(d))) {
  const key = fileKey(node)
  if (!key) { check(`${node}: widget-def 能解析出 file 字段名`, false); continue }

  // A. 空值时：上传入口在、选文件 → 上传 → 写回
  let written = null
  const el = mount(node, { params: { [key]: '' }, onParamsChange: (p) => { written = p } })
  const picker = findAllByTag(el, 'input').find((i) => i.type === 'file')
  check(`${node}: 渲染出 file input 且带 accept`, !!picker && !!picker.accept)
  check(`${node}: 有「📁 选择文件」按钮`, !!findByText(el, '📁 选择文件'))
  if (picker) {
    lastSentForm = null
    picker.files = [{ name: 'x.png', size: 2048 }]
    fire(picker, 'change')
    check(`${node}: 上传请求带上了文件`, !!lastSentForm && lastSentForm.entries[0][0] === 'file')
    check(`${node}: 上传成功把返回路径写回 ${key}`,
      !!written && written[key] === '~/flowx-input/uploads/T_x.png')
    check(`${node}: 上传后状态显示 ✅`, !!findByText(el, '✅ T_x.png'))
  }

  // B. 绑**上游节点输出**（paramSources.kind='node'）：只读来源标注，不出现上传入口
  const elW = mount(node, {
    params: { [key]: '{{ SomeNode.file }}' },
    paramSources: { [key]: { kind: 'node', nodeId: 'SomeNode', nodeName: 'SomeNode', field: 'file' } },
    onParamsChange: () => {},
  })
  check(`${node}: 绑上游节点输出时不出现上传入口（只读）`,
    !findByText(elW, '📁 选择文件') && !findAllByTag(elW, 'input').some((i) => i.type === 'file'))

  // B2. 绑**流水线参数**（paramSources.kind='workflow'）：仍要有上传入口 + 用参数当前值出预览
  //     （模板普遍写 image_path: "{{ Param.xxx }}"；一律只读会让用户没法换图 —— wf157 反馈）
  let writtenP = null
  const elP = mount(node, {
    params: { [key]: '{{ Param.src }}' },
    paramSources: { [key]: { kind: 'workflow', paramName: 'src', paramValue: '~/flowx-input/ecom/prod.png' } },
    onParamsChange: (pp) => { writtenP = pp },
  })
  const pickP = findAllByTag(elP, 'input').find((i) => i.type === 'file')
  check(`${node}: 绑流水线参数时仍有上传入口`, !!pickP && !!findByText(elP, '📁 选择文件'))
  check(`${node}: 绑流水线参数时提示参数名`,
    !!findByText(elP, '↳ 来自流水线参数 src（上传或直接填写将解除绑定，改为本节点直填路径）'))
  const prevP = findAllByTag(elP, 'img').find((i) => i.dataset.path)
  check(`${node}: 绑流水线参数时按参数当前值出预览图`, !!prevP)
  if (pickP) {
    lastSentForm = null
    pickP.files = [{ name: 'y.png', size: 1024 }]
    fire(pickP, 'change')
    check(`${node}: 绑流水线参数时上传可写回（解除绑定改为直填）`,
      !!writtenP && writtenP[key] === '~/flowx-input/uploads/T_x.png')
  }

  // C. 回放态（onParamsChange 缺省）：入口仍在但必须禁用
  const elR = mount(node, { params: { [key]: '~/flowx-input/a.png' } })
  const btnR = findByText(elR, '📁 选择文件')
  check(`${node}: 回放态上传按钮禁用（只读）`, !!btnR && btnR.disabled === true)
}

// ---------- 3. 未声明 kind:'file' 的节点不得出现上传入口 ----------
if (plainNode && targets.includes(plainNode)) {
  const elP = mount(plainNode, { params: {} })
  check(`${plainNode}（未声明 kind:'file'）无上传入口`,
    !findByText(elP, '📁 选择文件') && !findAllByTag(elP, 'input').some((i) => i.type === 'file'))
}

// ---------- 4. 弹层关闭：✕ / 点遮罩空白 / Esc ----------
function shot(node, prevFn, how) {
  const lb = prevFn()
  check(`${node}: ${how} 能打开放大弹窗`, !!lb)
  if (!lb) return
  if (how === '✕ 按钮') fire(findByText(lb, '✕'), 'click')
  else if (how === '点遮罩空白') fire(lb, 'click')
  else esc()
  check(`${node}: ${how} 能关闭放大弹窗（遮罩已移除）`, overlays().length === 0)
}
for (const node of fileNodes.filter((d) => targets.includes(d))) {
  const key = fileKey(node)
  const el = mount(node, { status: 'success', params: { [key]: '~/flowx-input/uploads/T_x.png' } })
  // 预览图 = 设了 dataset.path 的那个（预览帧 img 同 tag 但无 path）
  const prev = findAllByTag(el, 'img').find((i) => i.dataset.path)
  check(`${node}: 参数有本地路径时渲染出预览图`, !!prev && prev.style.display === 'block')
  if (!prev) continue
  const open = () => { document.body.children.length = 0; fire(prev, 'click'); return overlays()[0] }
  shot(node, open, '✕ 按钮')
  shot(node, open, '点遮罩空白')
  shot(node, open, 'Esc')
  open(); const twice = open()
  check(`${node}: 重复打开不叠加弹层`, !!twice && overlays().length === 1)
  fire(findByText(twice, '✕'), 'click')
  check(`${node}: 关闭后无残留遮罩`, overlays().length === 0)
}

// ---------- 5. 落盘结果媒体区（media*Output）同源同修 ----------
for (const [node, outKey, isVideo] of mediaNodes) {
  if (!targets.includes(node)) continue
  const el = mount(node, { status: 'success', outputs: { [outKey]: '~/flowx-output/out.png' } })
  const media = findAllByTag(el, isVideo ? 'video' : 'img').find((i) => i.dataset.path)
  check(`${node}: 结果媒体区渲染出文件`, !!media)
  if (!media) continue
  if (isVideo) {
    const exp = findByText(el, '⛶')
    document.body.children.length = 0
    fire(exp, 'click')
    check(`${node}: ⛶ 能打开视频弹层`, overlays().length === 1)
    fire(findByText(overlays()[0], '✕'), 'click')
    check(`${node}: ✕ 能关闭视频弹层`, overlays().length === 0)
  } else {
    document.body.children.length = 0
    fire(media, 'click')
    check(`${node}: 结果图能打开弹层`, overlays().length === 1)
    esc()
    check(`${node}: Esc 能关闭结果图弹层`, overlays().length === 0)
  }
}

// ---------- 6. 自定义 widget 通用回归（声明即被测）----------
// 扫：凡是 ui/widget-def.js 声明了 fields 的节点，都按声明逐项验控件与交互。
// 每个 kind 的期望 DOM：slider→input[type=range]、check→checkbox、select→select、
// curves→canvas、textarea→textarea、text→input[type=text]；compareInput→有对比图；
// replay→有「▶ 预览」按钮，且点击后把 outputs.__op_name **原样**付给节点级代理的
// /op（F1：版本化算子名，dev-plan §47.9 A7 —— 名字带版本，重放不会被后来注册的
// 同基名算子顶掉；这里用假名 x.y@v1 验"原样传递"这层契约，与真实版本号无关）。
// ⚠️ 解析失败的 bundle 必须**报 FAIL**，不能静默跳过：曾因为 widget-def 的 label
// 里嵌了未转义的单引号（'文本提示（多类用 ' . ' 分隔）'），产物语法非法——Studio 侧
// `import(bundleUrl)` 直接抛 SyntaxError，画布上该节点显示「UI 加载失败」，而这里
// 的 try/catch 把它过滤掉后整条门槛仍然全绿（流水线 135 事故）。
const fieldNodes = []
for (const d of Object.keys(defs)) {
  try { bundleSpec(d) } catch (e) {
    check(`${d}: ui/node-widget.js 可解析（base+widget-def 拼接产物语法合法）`, false)
    console.log(`      ${e.message}（跑 _tools/gen-widget-def.py 或修 widget-def.js 的引号）`)
    continue
  }
  fieldNodes.push(d)
}
if (!fieldNodes.length) check('至少有一个节点 widget 可解析', false)
function bundleSpec(node) {
  const file = path.join(NODES, node, 'ui', 'node-widget.js')
  const src = fs.readFileSync(file, 'utf8').replace(/^export default .*$/m, '')
  return new Function(src + '\nreturn { createNodeWidget, WIDGET_SPEC };')()
}
const KIND_DOM = {
  slider: (el) => findAllByTag(el, 'input').filter((i) => i.type === 'range').length,
  check: (el) => findAllByTag(el, 'input').filter((i) => i.type === 'checkbox').length,
  select: (el) => findAllByTag(el, 'select').length,
  curves: (el) => findAllByTag(el, 'canvas').length,
  textarea: (el) => findAllByTag(el, 'textarea').length,
  text: (el) => findAllByTag(el, 'input').filter((i) => i.type === 'text').length,
}
let mounted = 0
for (const node of fieldNodes) {
  if (only.length && !only.includes(node)) continue
  const spec = bundleSpec(node).WIDGET_SPEC
  const fields = spec.fields || []
  let el = null
  try { el = mount(node, { outputs: { __op_name: 'x.y@v1', __inputs_resolved: '{"image":{"id":"OBJ"}}' }, execution: { id: 624 } }) } catch (e) {
    check(`${node}: widget 能挂载（mount 不抛错）`, false)
    continue
  }
  mounted++
  check(`${node}: widget 能挂载（mount 不抛错）`, !!el)
  // 声明了几项，就至少渲染出对应的几个控件（advanced 项默认折叠时只要求不报错）
  for (const kind of Object.keys(KIND_DOM)) {
    const n = fields.filter((f) => f.kind === kind).length
    if (!n) continue
    const got = KIND_DOM[kind](el)
    check(`${node}: 声明 ${n} 个 kind:'${kind}' → 渲染 ${got}`, got >= n)
  }
  // 参数写回：动第一个滑杆，必须触发 onParamsChange
  const firstSlider = fields.find((f) => f.kind === 'slider')
  if (firstSlider) {
    let written = null
    const el2 = mount(node, { params: {}, onParamsChange: (p) => { written = p } })
    const ctl = findAllByTag(el2, 'input').filter((i) => i.type === 'range')[0]
    if (ctl) {
      ctl.value = String(Number(firstSlider.default) + Number(firstSlider.step || 1))
      fire(ctl, 'input'); fire(ctl, 'change')
      check(`${node}: 滑杆改动写回参数（${firstSlider.key}）`, !!written && written[firstSlider.key] !== undefined)
    }
  }
  // ▶预览重放（replay 声明）：按钮在 + 请求体原样带 __op_name + 走节点级代理 /op
  if (spec.replay && spec.replay.length) {
    const btn = findByText(el, '▶ 预览')
    check(`${node}: 声明 replay → 有「▶ 预览」按钮`, !!btn)
    if (btn) {
      let posted = null
      global.fetch = async (url, opt) => { posted = { url, body: JSON.parse(opt.body) }
        return { ok: true, status: 200, json: async () => ({ data: { outputs: { image: { id: 'NEW', type: 'IMAGE' } } } }) } }
      fire(btn, 'click')
      await new Promise((r) => setTimeout(r, 20))
      const dec = posted ? decodeURIComponent(String(posted.url)) : ''
      check(`${node}: 重放把 __op_name 原样打到 /op`, !!posted && posted.body.name === 'x.y@v1')
      check(`${node}: 重放走节点级代理（service-proxy&path=/op）`,
        dec.includes('service-proxy') && dec.includes('path=/op'))
      check(`${node}: 重放带解析后的上游输入`, !!posted && !!posted.body.inputs && !!posted.body.inputs.image)
      global.fetch = async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => '' })
    }
  }
}
console.log(`自定义 widget 节点：${mounted}/${fieldNodes.length} 个完成通用回归\n`)

// ---------- 7. 蒙版手绘：笔画抽稀 + 外置存储（strokes_ref）----------
// 事故背景：手绘时 pointermove 每帧 push 一点，一笔可达数千点，一次 workflow 定义
// 100KB+（workflow 113 执行 641：6 笔 7476 点 / 104KB，还随每次执行再复制一份到快照）。
// 抽稀 + 外置后：参数里只留短路径。
const paintNodes = Object.keys(defs).filter((d) => /strokesRefKey/.test(defs[d]))
for (const node of paintNodes) {
  const refKey = (defs[node].match(/strokesRefKey:\s*'([^']+)'/) || [])[1]
  const jsonKey = (defs[node].match(/strokesKey:\s*'([^']+)'/) || [])[1]
  check(`${node}: widget-def 能解析出 strokesRefKey/strokesKey`, !!refKey && !!jsonKey)

  // A. 点密集折线 → RDP 抽稀后再存（保留首尾、去掉近似共线的点）
  const simplify = bundleSpec(node).createNodeWidget.simplifyStrokes
  check(`${node}: 暴露抽稀函数（测试出口）`, typeof simplify === 'function')
  if (typeof simplify === 'function') {
    const line = []
    for (let x = 0; x <= 500; x += 1) line.push([x, 0])          // 501 点共线
    const lineOut = simplify([{ type: 'brush', size: 40, points: line }])[0].points
    check(`${node}: 共线 501 点抽稀成 2 点（首尾）`, lineOut.length === 2)
    check(`${node}: 抽稀保留首尾坐标`,
      lineOut[0][0] === 0 && lineOut[0][1] === 0
      && lineOut[1][0] === 500 && lineOut[1][1] === 0)

    const spike = line.slice()
    spike[250] = [250, 12]                                       // 真拐点（远大于 ε=1.5）
    const spikeOut = simplify([{ type: 'brush', size: 40, points: spike }])[0].points
    check(`${node}: 保留真拐点且大幅降点（501 → ≤8）`,
      spikeOut.length <= 8 && spikeOut.some((p) => p[1] === 12))
    check(`${node}: 抽稀结果为整数坐标`,
      spikeOut.every((p) => Number.isInteger(p[0]) && Number.isInteger(p[1])))

    const dup = []
    for (let i = 0; i < 200; i++) dup.push([5, 5])
    check(`${node}: 完全重复的 200 点去重成 1 点`,
      simplify([{ type: 'erase', size: 20, points: dup }])[0].points.length === 1)

    const rect = simplify([{ type: 'rect', x: 1.4, y: 2.6, w: 10.2, h: -3.6 }])
    check(`${node}: rect 抽稀后仍为 rect 且取整`,
      rect.length === 1 && rect[0].type === 'rect' && rect[0].x === 1 && rect[0].h === -4)
    check(`${node}: 未知类型笔画原样保留`, simplify([{ type: 'polygon', points: [[1, 2]] }]).length === 1)
    check(`${node}: 空笔画数组不报错`, simplify([]).length === 0 && simplify(null).length === 0)
  }

  // B. 参数里是外置路径 → 异步读回文件并渲染（缩略图提示反映笔数）
  const payload = JSON.stringify({
    strokes: [{ type: 'brush', size: 40, points: [[10, 10], [20, 20]] }],
    image_size: [768, 1280],
  })
  let fetched = ''
  global.fetch = async (u) => {
    fetched = decodeURIComponent(String(u))
    return { ok: true, status: 200, text: async () => payload }
  }
  const elRef = mount(node, {
    params: { [refKey]: '~/flowx-input/uploads/strokes/x.json' },
    onParamsChange: () => {},
  })
  await new Promise((r) => setTimeout(r, 10))
  check(`${node}: 外置路径经 /media/file 读取`, fetched.includes('/api/v1/media/file?') && fetched.includes('x.json'))
  check(`${node}: 读回后提示“1 笔”`,
    findAllByTag(elRef, 'span').some((s) => /1 笔/.test(s._text)))
  check(`${node}: 读回后允许编辑`, findByText(elRef, '✏️ 编辑蒙版').disabled === false)

  // C. 读不回来 → 明确报错并禁用编辑（不得用空笔画覆盖用户内容）
  global.fetch = async () => ({ ok: false, status: 404, text: async () => '' })
  const elErr = mount(node, {
    params: { [refKey]: '~/flowx-input/uploads/strokes/gone.json' },
    onParamsChange: () => {},
  })
  await new Promise((r) => setTimeout(r, 10))
  check(`${node}: 读取失败提示 ❌`, findAllByTag(elErr, 'span').some((s) => /读取失败/.test(s._text)))
  check(`${node}: 读取失败时禁用编辑`, findByText(elErr, '✏️ 编辑蒙版').disabled === true)

  // D. 编辑器保存：画几笔 → ✔ 完成 → 抽稀后上传成 JSON，写回 ref 并清空内联
  let written = null
  const elSave = mount(node, { params: {}, onParamsChange: (p) => { written = p } })
  document.body.children.length = 0
  fire(findByText(elSave, '✏️ 编辑蒙版'), 'click')
  const overlay = overlays()[0]
  check(`${node}: 点「✏️ 编辑蒙版」打开全屏编辑器`, !!overlay)
  if (overlay) {
    const areaEl = findAllByTag(overlay, 'div').find((d) => d._h.pointerdown)
    const pev = (x, y) => ({
      pointerId: 1, clientX: x, clientY: y,
      preventDefault() {}, stopPropagation() {},
      target: { closest: () => null },
    })
    if (areaEl) {
      fire(areaEl, 'pointerdown', pev(10, 10))
      for (let x = 11; x <= 60; x += 1) fire(areaEl, 'pointermove', pev(x, 10))
      fire(areaEl, 'pointerup', pev(60, 10))
    }
    lastSentForm = null
    fire(findByText(overlay, '✔ 完成'), 'click')
    await new Promise((r) => setTimeout(r, 20))
    const up = lastSentForm && lastSentForm.entries[0]
    check(`${node}: 保存走 /media/upload?subdir=strokes`,
      !!lastUploadUrl && lastUploadUrl.includes('/api/v1/media/upload') && lastUploadUrl.includes('subdir=strokes'))
    check(`${node}: 上传体是 JSON 文件`, !!up && up[0] === 'file' && up[1] && String(up[1].type).includes('json'))
    if (up && up[1] && up[1].text) {
      const body = JSON.parse(await up[1].text())
      check(`${node}: 上传的笔画已抽稀（50 点共线 → ≤3 点）`,
        (body.strokes[0].points || []).length <= 3)
      check(`${node}: 上传体带 image_size`, Array.isArray(body.image_size) && body.image_size.length === 2)
    }
    check(`${node}: 写回外置路径 ${refKey}`, !!written && !!written[refKey])
    check(`${node}: 同时清空旧的内联 ${jsonKey}`, !!written && written[jsonKey] === '')
  }
  global.fetch = async () => ({ ok: true, status: 200, json: async () => ({}), text: async () => '' })
}
if (!paintNodes.length) console.log('! 未发现声明 strokesRefKey 的 paint 节点（跳过第 7 节）')

console.log(fails ? `\n✗ ${fails} 项失败` : '\n✓ widget UI 回归全部通过')
process.exit(fails ? 1 : 0)
