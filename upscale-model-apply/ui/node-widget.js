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
 *     { key, label, kind:'model', modelType }                模型名下拉（经通用代理
 *       POST /api/v1/service-proxy 调第三方服务 GET /models/files，按 kind 过滤；
 *       API 路径语义由本节点生态自持）
 *   replay[]      （可选）▶预览按钮：经节点级通用代理重放算子（POST /op），
 *                 overrides 从这些字段当前值收集（slider 转 Number、check 转 Boolean）；
 *                 依赖 outputs 的 __op_name/__inputs_resolved
 *   compareInput  （可选）'image' 等输入键名：完成后显示前后对比滑块
 *                 （原图对象 id 从 outputs.__inputs_resolved 解析，经节点级通用代理
 *                 GET /images/{id} 拉取，结果图 = 预览区最终帧）
 *   previewEveryKey （可选）'preview_every'：显示"实时预览"开关（采样类节点）
 *   note          （可选）输出说明行
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

    let cur = props
    const refreshers = []
    const editable = () => typeof cur.onParamsChange === 'function'
    const setParam = (key, value) => {
      if (!editable()) return
      const params = { ...(cur.params || {}), [key]: String(value) }
      cur = { ...cur, params }
      cur.onParamsChange(params)
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
        const url = resolveParam('service_url')
        const tok = resolveParam('service_token') || ''
        if (!url || isWired(url)) {
          hint.textContent = 'service_url 未解析（绑定无当前值），无法拉取模型列表'
          return
        }
        try {
          const r = await fetch('/api/v1/inference/models-files', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ service_url: url, service_token: tok, method: 'GET', path: '/models/files' }),
          })
          if (!r.ok) throw new Error('HTTP ' + r.status)
          const data = await r.json()
          const files = ((data.data || data).files || []).filter((x) => x.kind === f.modelType)
          setOptions(files.map((x) => x.name))
          hint.textContent = files.length ? `${files.length} 个可选` : `无 kind=${f.modelType} 的模型`
          loaded = true
        } catch (e) {
          hint.textContent = '列表加载失败：' + e + '（可手动输入）'
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
      return inp
    }

    const controlOf = (f) => {
      if (f.kind === 'slider') return field(f.label, sliderControl(f))
      if (f.kind === 'check') return field('', checkControl(f))
      if (f.kind === 'select') return field(f.label, selectControl(f))
      if (f.kind === 'model') return field(f.label, modelControl(f))
      if (f.kind === 'textarea') return field(f.label, textareaControl(f))
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
    // ---- 节点级通用代理助手：第三方 API 路径语义由本节点生态自持 ----
    const nodeProxyUrl = (path, method) =>
      `/api/v1/executions/${cur.execution.id}/nodes/${encodeURIComponent(cur.nodeId)}/service-proxy?path=${encodeURIComponent(path)}&method=${method || 'GET'}`
    const resolvedInputs = () => {
      const raw = cur.outputs && cur.outputs.__inputs_resolved
      if (!raw) return null
      try { return typeof raw === 'string' ? JSON.parse(raw) : raw } catch { return null }
    }
    let replayImgUrl = ''    // ▶预览重放结果帧：覆盖 preview/对比区显示，直到新 preview 帧到达
    let replayBasePvUrl = '' // 重放时的 preview url（用于检测新帧到达）

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
      const box = h('div', 'position:relative;width:100%;border-radius:8px;overflow:hidden;background:rgba(0,0,0,0.3);user-select:none;touch-action:none')
      const afterImg = h('img', 'width:100%;display:block;pointer-events:none')   // 结果（底层全显）
      const beforeWrap = h('div', 'position:absolute;inset:0;overflow:hidden;pointer-events:none')
      const beforeImg = h('img', 'width:100%;display:block;pointer-events:none')  // 输入（顶层被裁）
      beforeWrap.append(beforeImg)
      const divider = h('div', 'position:absolute;top:0;bottom:0;width:2px;background:#22d3ee;cursor:ew-resize;box-shadow:0 0 6px rgba(34,211,238,0.8)')
      const knob = h('div', 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:18px;height:18px;border-radius:50%;background:#22d3ee;color:#0b1220;font-size:10px;display:flex;align-items:center;justify-content:center;pointer-events:none', '⇄')
      divider.append(knob)
      const tagL = h('span', 'position:absolute;left:4px;top:4px;font-size:9px;background:rgba(0,0,0,0.55);border-radius:4px;padding:1px 5px;color:rgba(255,255,255,0.85);pointer-events:none', '原图')
      const tagR = h('span', 'position:absolute;right:4px;top:4px;font-size:9px;background:rgba(0,0,0,0.55);border-radius:4px;padding:1px 5px;color:rgba(255,255,255,0.85);pointer-events:none', '结果')
      box.append(afterImg, beforeWrap, divider, tagL, tagR)
      const cmpHint = h('div', 'font-size:9px;color:rgba(255,255,255,0.35);margin-top:2px')
      cmpWrap.append(box, cmpHint)
      let ratio = 0.5
      const applyRatio = () => {
        beforeWrap.style.width = (ratio * 100) + '%'
        beforeImg.style.width = (100 / Math.max(ratio, 0.001)) + '%'
        divider.style.left = `calc(${(ratio * 100)}% - 1px)`
      }
      applyRatio()
      let dragging = false
      const move = (e) => {
        const rect = box.getBoundingClientRect()
        const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left
        ratio = Math.min(0.98, Math.max(0.02, x / rect.width))
        applyRatio()
      }
      divider.addEventListener('mousedown', (e) => { dragging = true; e.preventDefault() })
      window.addEventListener('mousemove', (e) => { if (dragging) move(e) })
      window.addEventListener('mouseup', () => { dragging = false })
      box.addEventListener('click', move)
      let cmpLoadedFor = ''
      refreshers.push(() => {
        // 结果帧 = 预览区最终帧（progress 1.0）或重放结果；原图对象 id 从
        // outputs.__inputs_resolved 解析，经节点级通用代理拉取
        const pv = cur.preview
        const effectiveUrl = replayImgUrl || (pv && pv.url)
        const done = effectiveUrl && (replayImgUrl || typeof pv.progress !== 'number' || pv.progress >= 1)
        if (!done || !cur.execution) { cmpWrap.style.display = 'none'; cmpLoadedFor = ''; return }
        if (cmpLoadedFor === effectiveUrl) { cmpWrap.style.display = 'block'; return }
        const resolved = resolvedInputs()
        const ref = resolved && resolved[spec.compareInput]
        const inId = ref && (typeof ref === 'string' ? ref : (ref.$id || ref.id || ''))
        if (!inId) {
          cmpWrap.style.display = 'none'
          cmpLoadedFor = ''
          return
        }
        cmpLoadedFor = effectiveUrl
        afterImg.src = effectiveUrl
        beforeImg.src = nodeProxyUrl('/images/' + inId, 'GET')
        beforeImg.onerror = () => { cmpHint.textContent = '原图加载失败（对象已驱逐或未执行过）' }
        afterImg.onload = () => {
          // 结果与原图尺寸不同（如放大 4x）：beforeImg 按 box 宽度缩放即可对比构图/细节
          cmpWrap.style.display = 'block'
          cmpHint.textContent = '拖动分割线对比 原图 / 结果'
        }
      })
    }

    // ---- ▶预览（op-replay 重放） ----
    const replayRow = h('div', 'display:none;align-items:center;margin-bottom:6px')
    if (spec.replay && spec.replay.length) {
      const btn = h('button', BTN_CSS + ';color:#34d399;border-color:rgba(52,211,153,0.4)', '▶ 预览')
      btn.title = '用上次执行的输入 + 当前参数重放该算子（秒级），结果直接刷新预览区'
      const hint = h('span', 'font-size:9px;color:rgba(255,255,255,0.4);margin-left:6px')
      const overrides = () => {
        const o = {}
        for (const key of spec.replay) {
          const f = (spec.fields || []).find((x) => x.key === key)
          const v = readParam(key, f && f.default)
          if (f && (f.kind === 'slider')) o[key] = Number(v)
          else if (f && f.kind === 'check') o[key] = (v === true || v === 'true' || v === '1' || v === 1)
          else o[key] = v
        }
        return o
      }
      btn.addEventListener('click', async () => {
        if (!cur.execution) { hint.textContent = '无执行实例，先运行一次'; return }
        const op = cur.outputs && cur.outputs.__op_name
        const resolved = resolvedInputs()
        if (!op || !resolved) { hint.textContent = '需先用当前版本节点执行一次'; return }
        btn.disabled = true
        hint.textContent = '重放中…'
        try {
          const r = await fetch(nodeProxyUrl('/op', 'POST'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: op, inputs: { ...resolved, ...overrides() } }),
          })
          if (!r.ok) {
            hint.textContent = r.status === 409 ? '需先用当前版本节点执行一次' : `失败 HTTP ${r.status}`
          } else {
            const resp = await r.json()
            const outs = ((resp.data || resp).outputs) || {}
            let imgId = ''
            for (const k of Object.keys(outs)) {
              const m = outs[k]
              if (m && m.type === 'IMAGE' && m.id) { imgId = m.id; break }
            }
            if (imgId) {
              replayBasePvUrl = (cur.preview && cur.preview.url) || ''
              replayImgUrl = nodeProxyUrl('/images/' + imgId, 'GET') + '&t=' + Date.now()
              prevImg.src = replayImgUrl
              prevWrap.style.display = 'block'
              cmpLoadedFor = '' // 强制对比区以重放结果帧重载
              refreshers.forEach((fn) => fn())
              hint.textContent = ''
            } else {
              hint.textContent = '重放完成（无图像输出）'
            }
          }
        } catch (e) {
          hint.textContent = '失败：' + e
        } finally {
          btn.disabled = false
        }
      })
      refreshers.push(() => { btn.disabled = false })
      replayRow.style.display = 'flex'
      replayRow.append(btn, hint)
    }

    // ---- 实时预览开关（采样类） ----
    const peRow = h('div', 'margin-bottom:6px;display:none')
    if (spec.previewEveryKey) {
      const k = spec.previewEveryKey
      const wrap = h('label', 'display:flex;align-items:center;gap:6px;font-size:10px;color:rgba(255,255,255,0.75);cursor:pointer')
      const cb = h('input', 'accent-color:#22d3ee')
      cb.type = 'checkbox'
      const cur_ = () => Number(readParam(k, 0)) > 0
      cb.checked = cur_()
      cb.addEventListener('change', () => setParam(k, cb.checked ? '5' : '0'))
      wrap.append(cb, document.createTextNode('实时预览（preview_every=5）'))
      refreshers.push(() => { cb.disabled = !editable(); cb.checked = cur_() })
      peRow.style.display = 'block'
      peRow.append(wrap)
    }

    // ---- 参数控件区 ----
    const controls = h('div', 'margin-bottom:6px')
    for (const f of spec.fields || []) controls.append(controlOf(f))

    // ---- 输出区 ----
    const outBox = h('div', 'color:rgba(255,255,255,0.45);word-break:break-all')

    el.append(header, prevWrap, cmpWrap, controls, replayRow, peRow, outBox)
    if (spec.note) el.append(h('div', 'font-size:9px;color:rgba(255,255,255,0.3);margin-top:4px', spec.note))

    let lastPreviewUrl = ''
    function render(p) {
      cur = p
      dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
      statusLabel.textContent = p.status
      const pv = p.preview
      if (pv && pv.url) {
        prevWrap.style.display = 'block'
        // 新一轮 preview 帧（url 变化）到达时解除重放结果覆盖
        if (replayImgUrl && pv.url !== replayBasePvUrl) { replayImgUrl = ''; replayBasePvUrl = '' }
        const effective = replayImgUrl || pv.url
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
        // 无 preview 帧信号的老执行器：回退展示 emit 的 thumb_b64 结果缩略图
        const thumb = (p.outputs || {}).thumb_b64
        if (thumb && typeof thumb === 'string' && thumb.startsWith('data:')) {
          prevWrap.style.display = 'block'
          progOuter.style.display = 'none'
          stopBtn.style.display = 'none'
          if (thumb !== lastPreviewUrl) {
            lastPreviewUrl = thumb
            prevImg.src = thumb
          }
        } else {
          prevWrap.style.display = 'none'
          lastPreviewUrl = ''
        }
      }
      const o = p.outputs || {}
      outBox.textContent = ''
      const entries = Object.entries(o).filter(([k]) => !k.startsWith('__'))
      if (entries.length === 0) {
        outBox.textContent = '等待执行…'
      } else {
        for (const [k, v] of entries) {
          outBox.append(h('div', '', `${k}: ${String(v).slice(0, 80)}`))
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
}

// ---- 节点专属定义（widget-def.js，由 build-widget.py 拼接，请勿直接编辑本文件）----
const WIDGET_SPEC = {
  icon: '🔍',
  title: '模型放大',
  fields: [
    { key: 'service_url', label: '推理服务 service_url', kind: 'text', mono: true },
    { key: 'upscale_model', label: 'upscale_model（放大模型加载输出）', kind: 'text', mono: true },
    { key: 'image', label: 'image（待放大图）', kind: 'text', mono: true },
    { key: 'tile', label: '分块 tile（0=整图）', kind: 'slider', min: 0, max: 1024, step: 64, default: 0 },
    { key: 'service_token', label: 'service_token（可空）', kind: 'text', mono: true },
  ],
  replay: ['tile'],
  compareInput: 'image',
  note: '完成后显示 原图/结果 前后对比；tile>0 分块防大分辨率爆显存',
}

export default createNodeWidget(WIDGET_SPEC)
