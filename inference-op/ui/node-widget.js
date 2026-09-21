/**
 * inference-op 画布组件：通用算子参数控件（op_name / inputs_json / emit_keys）+ 输出端口。
 * 契约：mount(el, props) => { update(props), unmount() }（ui.apiVersion: 1）
 *   props.params — 当前 config.params 绑定（模板值只读展示）
 *   props.paramSources — 参数绑定来源标注（可选）：workflow=流水线参数(含当前值) / node=上游节点(含显示名与运行时值) / literal=字面值
 *   props.onParamsChange(params) — 全量写回该节点参数（回放态缺省 = 控件只读）
 *
 * inputs_json 可读性增强（方案 A）：格式化按钮（pretty-print 多行展示）、
 * JSON 语法错误红框 + 错误行、字面量摘要区——把 text/seed/steps 等字面量
 * 从单行 JSON 里提取成高亮只读摘要（长文本如提示词整块换行展示）。
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
const BTN_CSS = 'font-size:10px;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.15);border-radius:6px;padding:2px 8px;color:rgba(255,255,255,0.75);cursor:pointer'

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

  // inputs_json 专用控件（方案 A 可读性增强）：
  // - 格式化按钮：pretty-print 多行展示（{{ }} 模板在字符串值内，不受影响）
  // - JSON 语法错误：红框 + 错误行（不阻断写回，仅视觉警告）
  // - 字面量摘要：解析后把字面量 key 提取成高亮只读摘要行；长文本（提示词等）
  //   整块换行展示——关键内容不再埋在单行 JSON 里；绑定值（$id/{{ }}）淡显
  const DIM_CSS = 'color:rgba(165,180,252,0.7);word-break:break-all;font-family:ui-monospace,Menlo,monospace;font-size:9px'
  const inputsJsonField = () => {
    // 注意：inputs_json 不做 isWired 整体绑定回退——它结构上是 JSON 文本，
    // {{ }} 模板合法地出现在字符串值内部（如 {"$id": "{{ X.clip }}"}），
    // 一律走 JSON 编辑器 + 摘要（模板值在摘要区淡显）；JSON.parse 不受影响。
    const wrap = h('div', '')
    const ta = h('textarea', INPUT_CSS + ';font-family:ui-monospace,Menlo,monospace;resize:vertical;min-height:52px')
    ta.rows = 3
    ta.value = (cur.params || {}).inputs_json !== undefined ? (cur.params || {}).inputs_json : ''
    ta.placeholder = '{"key": "value"}'
    ta.spellcheck = false
    const btnRow = h('div', 'display:flex;gap:6px;margin-top:3px')
    const fmtBtn = h('button', 'font-size:10px;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.15);border-radius:6px;padding:2px 8px;color:rgba(255,255,255,0.75);cursor:pointer', '格式化')
    const errLine = h('div', 'display:none;font-size:9px;color:#fb7185;margin-top:2px;word-break:break-all')
    const summary = h('div', 'margin-top:4px')
    const parse = () => {
      const v = ta.value.trim()
      if (!v) return { ok: true, obj: {} }
      try { return { ok: true, obj: JSON.parse(v) } } catch (e) {
        return { ok: false, err: String((e && e.message) || e) }
      }
    }
    const chip = (key, valText, valStyle) => {
      const row = h('div', 'display:flex;gap:6px;font-size:10px;line-height:1.5;padding:2px 6px;background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.15);border-radius:6px;margin-bottom:3px')
      row.append(h('span', 'color:rgba(52,211,153,0.85);font-family:ui-monospace,Menlo,monospace;flex-shrink:0', key + ':'))
      row.append(h('span', valStyle || 'color:rgba(255,255,255,0.85);word-break:break-all', valText))
      return row
    }
    const renderSummary = () => {
      summary.textContent = ''
      const r = parse()
      if (!r.ok) {
        errLine.style.display = 'block'
        errLine.textContent = '⚠ JSON 语法错误：' + r.err
        ta.style.borderColor = 'rgba(251,113,133,0.7)'
        return
      }
      errLine.style.display = 'none'
      ta.style.borderColor = 'rgba(255,255,255,0.14)'
      const obj = r.obj
      if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return
      for (const [k, v] of Object.entries(obj)) {
        if (v !== null && typeof v === 'object') {
          summary.append(chip(k, v.$id !== undefined
            ? '🔗 ' + String(v.$id) : JSON.stringify(v).slice(0, 120), DIM_CSS))
        } else if (typeof v === 'string' && v.indexOf('{{') >= 0) {
          summary.append(chip(k, v, DIM_CSS))
        } else if (typeof v === 'string' && v.length > 24) {
          // 长文本（提示词等）整块展示
          const box = h('div', 'font-size:10px;line-height:1.5;padding:3px 6px;background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.25);border-radius:6px;margin-bottom:3px')
          box.append(h('span', 'color:rgba(251,191,36,0.9);font-family:ui-monospace,Menlo,monospace', k + ': '))
          box.append(h('span', 'color:rgba(255,255,255,0.9);word-break:break-word;white-space:pre-wrap', v))
          summary.append(box)
        } else {
          summary.append(chip(k, JSON.stringify(v)))
        }
      }
    }
    ta.addEventListener('input', renderSummary)
    ta.addEventListener('change', () => setParam('inputs_json', ta.value))
    fmtBtn.addEventListener('click', () => {
      const r = parse()
      if (!r.ok) {
        errLine.style.display = 'block'
        errLine.textContent = '⚠ 无法格式化：' + r.err
        return
      }
      ta.value = JSON.stringify(r.obj, null, 2)
      if (editable()) setParam('inputs_json', ta.value)
      renderSummary()
    })
    btnRow.append(fmtBtn)
    refreshers.push(() => {
      ta.disabled = !editable()
      fmtBtn.disabled = !editable()
      ta.style.opacity = editable() ? '1' : '0.55'
      if (document.activeElement !== ta) {
        const nv = (cur.params || {}).inputs_json
        ta.value = nv !== undefined ? nv : ''
      }
      renderSummary()
    })
    wrap.append(ta, btnRow, errLine, summary)
    return field('入参 inputs_json', wrap)
  }

  const header = h('div', 'display:flex;align-items:center;gap:6px;margin-bottom:6px')
  const dot = h('span', 'width:8px;height:8px;border-radius:50%;flex-shrink:0')
  const title = h('strong', 'font-size:12px', '🧩 通用算子')
  const statusLabel = h('span', 'color:rgba(255,255,255,0.35);margin-left:auto')
  header.append(dot, title, statusLabel)

  // ---------- 预览/结果区（props.preview：执行中过程帧 + 完成后结果图） ----------
  // 帧经 Studio preview-frame 中转（同源 cookie 认证），点击放大 overlay。
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
  prevImg.addEventListener('error', () => {
    // 帧过期（推理侧对象被驱逐/Studio 来源映射过期）→ 隐藏预览区
    prevWrap.style.display = 'none'
  })
  const progOuter = h('div', 'height:4px;background:rgba(255,255,255,0.08);border-radius:2px;margin-top:4px;overflow:hidden')
  const progInner = h('div', 'height:100%;background:#22d3ee;width:0%;transition:width .3s')
  progOuter.append(progInner)
  const prevRow = h('div', 'display:flex;align-items:center;gap:6px;margin-top:4px')
  const prevInfo = h('span', 'font-size:9px;color:rgba(255,255,255,0.4);flex:1;word-break:break-all')
  // ---- 节点级通用代理助手：第三方 API 路径语义由本节点生态自持 ----
  const nodeProxyUrl = (path, method) =>
    `/api/v1/executions/${cur.execution.id}/nodes/${encodeURIComponent(cur.nodeId)}/service-proxy?path=${encodeURIComponent(path)}&method=${method || 'GET'}`
  const resolvedInputs = () => {
    const raw = cur.outputs && cur.outputs.__inputs_resolved
    if (!raw) return null
    try { return typeof raw === 'string' ? JSON.parse(raw) : raw } catch { return null }
  }

  const stopBtn = h('button', BTN_CSS + ';color:#fb7185;border-color:rgba(251,113,133,0.4)', '■ 中断')
  stopBtn.title = '中断该节点在第三方服务上的运行中任务（经节点级通用代理 POST /interrupt）'
  stopBtn.addEventListener('click', async () => {
    if (!cur.execution) { prevInfo.textContent = '无执行实例'; return }
    const jobId = cur.preview && cur.preview.jobId
    if (!jobId) { prevInfo.textContent = '节点未上报任务标识（旧节点包），无法中断'; return }
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

  // ---------- 算子专属快捷控件（预处理调参 + 即时预览 / 采样预览开关） ----------
  // 控件值与 inputs_json 双向同步（合并写回，不动其他键）；预览按钮经节点级
  // 通用代理 POST /op：用上次执行的解析后入参（outputs.__inputs_resolved）
  // + 当前 overrides 重放算子，结果图直接刷新上方预览区。
  const opExtras = h('div', 'margin-bottom:6px')
  const readInput = (k, dflt) => {
    try {
      const o = JSON.parse((cur.params || {}).inputs_json || '{}')
      return o[k] !== undefined ? o[k] : dflt
    } catch { return dflt }
  }
  const mergeInputsJson = (patch) => {
    let obj = {}
    try { obj = JSON.parse((cur.params || {}).inputs_json || '{}') } catch { return }
    Object.assign(obj, patch)
    setParam('inputs_json', JSON.stringify(obj))
  }
  const replayBtn = (getOverrides) => {
    const btn = h('button', BTN_CSS + ';color:#34d399;border-color:rgba(52,211,153,0.4)', '▶ 预览')
    btn.title = '用上次执行的输入 + 当前参数重放该算子（秒级），结果直接刷新预览区'
    const hint = h('span', 'font-size:9px;color:rgba(255,255,255,0.4);margin-left:6px')
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
          body: JSON.stringify({ name: op, inputs: { ...resolved, ...getOverrides() } }),
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
            prevImg.src = nodeProxyUrl('/images/' + imgId, 'GET') + '&t=' + Date.now()
            prevWrap.style.display = 'block'
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
    const row = h('div', 'display:flex;align-items:center;margin-top:4px')
    row.append(btn, hint)
    return row
  }
  const sliderRow = (label, key, min, max, dflt) => {
    const wrap = h('div', 'margin-bottom:4px')
    const val = h('span', 'color:rgba(255,255,255,0.7);font-family:ui-monospace,Menlo,monospace', String(readInput(key, dflt)))
    const lab = h('label', LABEL_CSS + ';display:flex;justify-content:space-between', '')
    lab.append(document.createTextNode(label), val)
    const rng = h('input', 'width:100%;accent-color:#22d3ee')
    rng.type = 'range'; rng.min = min; rng.max = max; rng.step = 1
    rng.value = readInput(key, dflt)
    rng.disabled = !editable()
    rng.addEventListener('input', () => { val.textContent = rng.value })
    rng.addEventListener('change', () => mergeInputsJson({ [key]: Number(rng.value) }))
    wrap.append(lab, rng)
    refreshers.push(() => {
      rng.disabled = !editable()
      const nv = readInput(key, dflt)
      if (document.activeElement !== rng) { rng.value = nv; val.textContent = String(nv) }
    })
    return wrap
  }
  const checkRow = (label, key, dflt) => {
    const wrap = h('label', 'display:flex;align-items:center;gap:6px;font-size:10px;color:rgba(255,255,255,0.75);margin-bottom:4px;cursor:pointer')
    const cb = h('input', 'accent-color:#22d3ee')
    cb.type = 'checkbox'
    cb.checked = Boolean(readInput(key, dflt))
    cb.disabled = !editable()
    cb.addEventListener('change', () => mergeInputsJson({ [key]: cb.checked }))
    wrap.append(cb, document.createTextNode(label))
    refreshers.push(() => {
      cb.disabled = !editable()
      cb.checked = Boolean(readInput(key, dflt))
    })
    return wrap
  }
  const SAMPLE_OPS = ['sample', 'sd3.sample', 'video.sample', 'video.sample_latent']
  let extrasFor = null
  const buildOpExtras = () => {
    const op = ((cur.params || {}).op_name || '').trim()
    if (op === extrasFor) return
    extrasFor = op
    opExtras.textContent = ''
    if (op === 'preprocess.canny') {
      opExtras.append(
        sliderRow('低阈值 low_threshold', 'low_threshold', 0, 255, 100),
        sliderRow('高阈值 high_threshold', 'high_threshold', 0, 255, 200),
        replayBtn(() => ({
          low_threshold: readInput('low_threshold', 100),
          high_threshold: readInput('high_threshold', 200),
        })),
      )
    } else if (op === 'preprocess.openpose') {
      opExtras.append(
        checkRow('手部 include_hand', 'include_hand', false),
        checkRow('面部 include_face', 'include_face', false),
        replayBtn(() => ({
          include_hand: Boolean(readInput('include_hand', false)),
          include_face: Boolean(readInput('include_face', false)),
        })),
      )
    } else if (SAMPLE_OPS.indexOf(op) >= 0) {
      // 实时预览开关：直接映射 inputs_json 的 preview_every（5/0），不复用 checkRow
      //（checkRow 会以 label 键名写回，产生 __pe_toggle 脏键）
      const wrap = h('label', 'display:flex;align-items:center;gap:6px;font-size:10px;color:rgba(255,255,255,0.75);margin-bottom:4px;cursor:pointer')
      const cb = h('input', 'accent-color:#22d3ee')
      cb.type = 'checkbox'
      cb.checked = readInput('preview_every', 0) > 0
      cb.disabled = !editable()
      cb.addEventListener('change', () => mergeInputsJson({ preview_every: cb.checked ? 5 : 0 }))
      wrap.append(cb, document.createTextNode('实时预览（preview_every=5）'))
      refreshers.push(() => {
        cb.disabled = !editable()
        cb.checked = readInput('preview_every', 0) > 0
      })
      opExtras.append(wrap)
    }
  }

  const controls = h('div', 'margin-bottom:6px')
  controls.append(
    field('算子 op_name', textControl('op_name', 'upscale / latent.empty / …', true)),
    inputsJsonField(),
    field('输出键 emit_keys（逗号分隔）', textControl('emit_keys', 'latent,image', true)),
  )

  const outBox = h('div', 'color:rgba(255,255,255,0.45);word-break:break-all')

  el.append(header, prevWrap, controls, opExtras, outBox)

  let lastPreviewUrl = ''
  function render(p) {
    cur = p
    dot.style.background = STATUS_COLORS[p.status] || STATUS_COLORS.idle
    statusLabel.textContent = p.status
    // 预览/结果区：preview.url 变化（SSE node_preview 带新时间戳）时刷新；
    // progress<1 显示进度条与中断按钮，完成（1.0）后保留最后一帧即结果图
    const pv = p.preview
    if (pv && pv.url) {
      prevWrap.style.display = 'block'
      if (pv.url !== lastPreviewUrl) {
        lastPreviewUrl = pv.url
        prevImg.src = pv.url
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
      prevWrap.style.display = 'none'
      lastPreviewUrl = ''
    }
    const o = p.outputs || {}
    outBox.textContent = ''
    const entries = Object.entries(o).filter(([k]) => !k.startsWith('__'))
    if (entries.length === 0) {
      outBox.textContent = '等待执行…'
    } else {
      for (const [k, v] of entries) {
        const row = h('div')
        row.textContent = `${k}: ${String(v).slice(0, 80)}`
        outBox.append(row)
      }
    }
    buildOpExtras()
    refreshers.forEach((f) => f())
  }

  render(props)

  return {
    update(next) { render(next) },
    unmount() { el.textContent = '' },
  }
}
