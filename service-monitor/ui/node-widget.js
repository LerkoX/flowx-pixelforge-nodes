/**
 * service-monitor 自定义 UI 组件（module 模式，ui.apiVersion: 1）
 *
 * 推理服务监控面板：实时显示 monitor 页关键信息（GPU / VRAM / 常驻模型 /
 * 租约 / 任务队列）。数据两条路：
 *  ① 实时轮询（默认）：经设计期通用代理 POST /api/v1/service-proxy 每 3s
 *     拉 /health + /gpu + /leases + /jobs，无需运行流水线。地址/token 从
 *     props.paramSources（workflow.paramValue / node.runtimeValue）或字面量
 *     参数解析，优先 service_url_host（宿主机可达），回退 service_url。
 *  ② 执行快照（回放兜底）：地址解析不出（未绑定等）但 props.outputs.snapshot
 *     存在时，渲染该次运行留存的快照并标注「执行快照」。
 *
 * 契约：默认导出 mount(el, props) => { update(props), unmount() }
 */

const POLL_MS = 3000

function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function fmtMB(mb) {
  if (mb == null || isNaN(Number(mb))) return '—'
  mb = Number(mb)
  return mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB' : Math.round(mb) + ' MB'
}

function fmtSec(s) {
  if (s == null || isNaN(Number(s))) return '—'
  s = Number(s)
  if (s < 60) return s.toFixed(0) + 's'
  if (s < 3600) return Math.floor(s / 60) + 'm' + Math.floor(s % 60) + 's'
  return Math.floor(s / 3600) + 'h' + Math.floor((s % 3600) / 60) + 'm'
}

function shortId(s) {
  s = String(s || '')
  return s.length > 20 ? s.slice(0, 12) + '…' + s.slice(-4) : s
}

function fmtTime(ts) {
  const d = new Date(ts * 1000)
  const p = n => String(n).padStart(2, '0')
  return p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds())
}

export default function mount(el, props) {
  el.style.cssText = [
    'font-family: system-ui, sans-serif',
    'background: rgba(255,255,255,0.04)',
    'border: 1px solid rgba(255,255,255,0.08)',
    'border-radius: 12px',
    'padding: 10px',
    'font-size: 11px',
    'line-height: 1.6',
    'color: rgba(255,255,255,0.85)',
    'overflow: auto',
    'box-sizing: border-box',
    'height: 100%',
  ].join(';')

  // ---- DOM 骨架 ----
  const header = document.createElement('div')
  header.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:6px'
  const dot = document.createElement('span')
  dot.style.cssText = 'width:8px;height:8px;border-radius:50%;flex-shrink:0;background:#94a3b8'
  const title = document.createElement('strong')
  title.textContent = '📊 推理服务监控'
  title.style.fontSize = '12px'
  const timeLabel = document.createElement('span')
  timeLabel.style.cssText = 'color:rgba(255,255,255,0.35);margin-left:auto;font-size:10px'
  header.append(dot, title, timeLabel)

  const notice = document.createElement('div')
  notice.style.cssText = 'display:none;margin-bottom:6px;padding:4px 6px;border-radius:6px;font-size:10px'

  const body = document.createElement('div')
  el.append(header, notice, body)

  // ---- 状态 ----
  let cur = props || {}
  let timer = null
  let inflight = false
  let liveSnap = null      // 最近一次实时轮询得到的快照（normalize 后）
  let lastErr = null       // 最近一次轮询错误（保留旧数据 + 黄条提示）

  // ---- 连接解析：paramSources 优先（绑定值的真实值只在这里），字面量兜底 ----
  function resolveParam(p, key) {
    const src = (p.paramSources || {})[key]
    if (src) {
      if (src.kind === 'workflow' && src.paramValue !== undefined && src.paramValue !== '') return src.paramValue
      if (src.kind === 'node' && src.runtimeValue !== undefined && src.runtimeValue !== '') return src.runtimeValue
    }
    const raw = (p.params || {})[key]
    if (raw != null && !String(raw).includes('{{')) return String(raw)
    return ''
  }

  function resolveConn() {
    const base = (resolveParam(cur, 'service_url_host') || resolveParam(cur, 'service_url') || '')
      .replace(/\/+$/, '')
    return { base, tok: resolveParam(cur, 'service_token') || '' }
  }

  async function proxyGet(base, tok, path) {
    const r = await fetch('/api/v1/service-proxy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        service_url: base, service_token: tok,
        method: 'GET', path, timeout_sec: 10,
      }),
    })
    const text = await r.text()
    let data = null
    try { data = JSON.parse(text) } catch { /* 上游非 JSON */ }
    if (!r.ok) {
      const detail = (data && (data.message || data.detail || data.error)) || text.slice(0, 120)
      throw new Error('HTTP ' + r.status + (detail ? '：' + detail : ''))
    }
    return data
  }

  // 把四个接口的原始响应整成与 main.py snapshot 同构的对象（渲染只认这个形态）
  function normalize(h, g, lz, jb) {
    h = h || {}; g = g || {}; lz = lz || {}; jb = jb || {}
    const gpu = (g.available && g.gpus && g.gpus[0]) || {}
    const leases = lz.leases || []
    const holderOf = {}
    leases.forEach(l => (l.keys || []).forEach(k => { holderOf[k] = l.lease }))
    const jobs = jb.jobs || []
    const active = jobs
      .filter(j => j.status === 'running' || j.status === 'pending')
      .sort((a, b) => (a.status === 'running' ? 0 : 1) - (b.status === 'running' ? 0 : 1))
    return {
      ts: Date.now() / 1000,
      gpu_name: h.gpu_name || '',
      util: gpu.utilization_percent ?? null,
      temp: gpu.temperature_c ?? null,
      power: gpu.power_w ?? null,
      power_limit: gpu.power_limit_w ?? null,
      fan: gpu.fan_percent ?? null,
      throttle: (gpu.throttle_reasons || []).filter(r => r !== 'gpu_idle'),
      vram: {
        total: h.vram_total_mb ?? null,
        alloc: h.vram_process_allocated_mb ?? null,
        reserved: h.vram_process_reserved_mb ?? null,
        cache: h.vram_reclaimable_mb ?? null,
        driver_free: h.vram_driver_free_mb ?? null,
        headroom: h.vram_headroom_mb ?? null,
      },
      oversubscribed: !!h.vram_oversubscribed,
      saturated: !!h.vram_saturated,
      resident_max: h.max_resident_models ?? null,
      residents: (h.resident_details || []).slice(0, 20).map(d => ({
        key: d.key, arch: d.arch, vram_mb: d.est_vram_mb,
        idle: d.idle_seconds, pinned: !!d.pinned, holder: holderOf[d.key] || null,
      })),
      leases: leases.slice(0, 20).map(l => ({
        lease: l.lease, keys: (l.keys || []).length, idle: l.idle_seconds,
      })),
      queue: jb.queue || null,
      jobs: active.slice(0, 8).map(j => ({
        id: shortId(j.id), ops: j.ops || [], status: j.status,
        queue_pos: j.queue_pos ?? null,
        percent: j.progress ? j.progress.percent ?? null : null,
      })),
    }
  }

  async function poll() {
    if (inflight) return
    const { base, tok } = resolveConn()
    if (!base) { render(); return }
    inflight = true
    try {
      const [h, g, lz, jb] = await Promise.all([
        proxyGet(base, tok, '/health'),
        proxyGet(base, tok, '/gpu'),
        proxyGet(base, tok, '/leases'),
        proxyGet(base, tok, '/jobs'),
      ])
      liveSnap = normalize(h, g, lz, jb)
      lastErr = null
    } catch (e) {
      lastErr = (e && e.message) || String(e)
    }
    inflight = false
    render()
  }

  // ---- 各区块渲染（返回 HTML 字符串） ----
  const DIM = 'color:rgba(255,255,255,0.4)'
  const WARN = 'color:#fbbf24'

  function secTitle(t, right) {
    return `<div style="margin:8px 0 3px;font-size:10px;letter-spacing:.05em;${DIM}">` +
      `${esc(t)}${right ? ` <span style="margin-left:4px">${right}</span>` : ''}</div>`
  }

  function kvRows(rows) {
    return rows.map(([k, v, sub, warn]) =>
      `<div style="display:flex;justify-content:space-between;gap:6px">` +
      `<span style="${DIM}">${esc(k)}</span>` +
      `<span style="text-align:right;${warn ? WARN : ''}">${esc(v)}` +
      (sub ? ` <span style="${DIM};font-size:10px">${esc(sub)}</span>` : '') +
      `</span></div>`).join('')
  }

  function renderGpu(s) {
    const rows = [
      ['SM 利用率', s.util != null ? s.util + ' %' : '—'],
      ['温度', s.temp != null ? s.temp + ' ℃' : '—', '', s.temp != null && s.temp >= 83],
      ['功耗', s.power != null ? s.power + ' W' : '—',
        s.power_limit != null ? '上限 ' + s.power_limit + ' W' : ''],
      ['风扇', s.fan != null ? s.fan + ' %' : '—'],
      ['限频', s.throttle && s.throttle.length ? s.throttle.join(', ') : '无', '',
        !!(s.throttle && s.throttle.length)],
    ]
    return secTitle('GPU', s.gpu_name ? `｜ ${esc(s.gpu_name)}` : '') +
      kvRows(rows)
  }

  function renderVram(s) {
    const v = s.vram || {}
    const total = v.total || 0
    if (!total) return secTitle('VRAM') + `<div style="${DIM}">CUDA 不可用</div>`
    const alloc = v.alloc || 0
    const cache = v.cache != null ? v.cache : Math.max(0, (v.reserved || alloc) - alloc)
    const usedNvml = v.driver_free != null ? Math.max(0, total - v.driver_free) : null
    const other = usedNvml != null ? Math.max(0, usedNvml - (v.reserved || alloc)) : 0
    const free = usedNvml != null ? Math.max(0, total - usedNvml) : Math.max(0, total - (v.reserved || 0))
    const pct = x => Math.max(0, Math.min(100, (x / total) * 100))
    const w0 = pct(alloc), w1 = pct(cache), w2 = pct(other)
    const bar =
      `<div style="position:relative;height:10px;border-radius:5px;overflow:hidden;` +
      `background:#21262d;margin:2px 0 4px">` +
      `<div style="position:absolute;left:0;top:0;bottom:0;width:${w0}%;background:#3b82f6"></div>` +
      `<div style="position:absolute;left:${w0}%;top:0;bottom:0;width:${w1}%;background:rgba(31,111,235,0.45)"></div>` +
      `<div style="position:absolute;left:${w0 + w1}%;top:0;bottom:0;width:${w2}%;background:#6b7280"></div>` +
      `</div>`
    const legend =
      `<div style="display:flex;flex-wrap:wrap;gap:4px 10px;font-size:10px;${DIM}">` +
      `<span><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:#3b82f6;margin-right:3px"></i>已分配 ${fmtMB(alloc)}</span>` +
      `<span><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:rgba(31,111,235,0.45);margin-right:3px"></i>缓存 ${fmtMB(cache)}</span>` +
      (w2 > 0.5 ? `<span><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:#6b7280;margin-right:3px"></i>其它 ${fmtMB(other)}</span>` : '') +
      `<span>空闲 ${fmtMB(free)}</span></div>`
    const rows = [
      ['已用 / 总量', fmtMB(usedNvml != null ? usedNvml : v.reserved) + ' / ' + fmtMB(total)],
      ['物理余量', fmtMB(v.headroom), '闸门口径'],
      ['超订', s.oversubscribed ? '是（退化）' : '否', s.saturated ? '显存饱和' : '', s.oversubscribed],
    ]
    return secTitle('VRAM') + bar + legend + kvRows(rows)
  }

  function renderModels(s) {
    const rows = s.residents || []
    const head = secTitle('常驻模型', `${rows.length} / ${s.resident_max ?? '?'} 个`)
    if (!rows.length) return head + `<div style="${DIM}">无常驻模型</div>`
    return head + rows.map(d => {
      const chips =
        (d.pinned ? `<span style="color:#34d399;font-size:10px">pin </span>` : '') +
        (d.holder ? `<span style="color:#22d3ee;font-size:10px" title="租约 ${esc(d.holder)}">租约 ${esc(shortId(d.holder))}</span>` : '')
      return `<div style="display:flex;justify-content:space-between;gap:6px">` +
        `<span class="mono" style="font-family:ui-monospace,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(d.key)}">${esc(shortId(d.key))}</span>` +
        `<span style="flex-shrink:0;text-align:right">${fmtMB(d.vram_mb)} <span style="${DIM};font-size:10px">空闲 ${fmtSec(d.idle)}</span> ${chips}</span></div>`
    }).join('')
  }

  function renderLeases(s) {
    const ls = s.leases || []
    const head = secTitle('执行租约', ls.length + ' 个活跃')
    if (!ls.length) return head + `<div style="${DIM}">无活跃租约</div>`
    return head + ls.map(l =>
      `<div style="display:flex;justify-content:space-between;gap:6px">` +
      `<span style="font-family:ui-monospace,monospace" title="${esc(l.lease)}">${esc(shortId(l.lease))}</span>` +
      `<span style="${DIM}">持 ${l.keys} 个模型 ｜ 空闲 ${fmtSec(l.idle)}</span></div>`).join('')
  }

  function renderJobs(s) {
    const q = s.queue
    const qtxt = q
      ? (q.running ? `worker 忙：${esc(shortId(q.running_title || ''))}（${fmtSec(q.running_seconds)}）` : 'worker 空闲') +
        (q.pending ? ` · 排队 ${q.pending}（队头等 ${fmtSec(q.head_wait_seconds)}）` : '')
      : ''
    const head = secTitle('任务队列', `${(s.jobs || []).length} 个进行中`)
    const js = s.jobs || []
    const list = js.length ? js.map(j => {
      const st = j.status === 'running'
        ? `<span style="color:#22d3ee">运行${j.percent != null ? ' ' + Math.round(j.percent) + '%' : ''}</span>`
        : `<span style="color:#fbbf24">排队${j.queue_pos ? ' #' + j.queue_pos : ''}</span>`
      return `<div style="display:flex;justify-content:space-between;gap:6px">` +
        `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc((j.ops || []).join(' → '))}">${esc((j.ops || []).join(' → ') || j.id)}</span>` +
        `<span style="flex-shrink:0">${st}</span></div>`
    }).join('') : `<div style="${DIM}">没有进行中的任务</div>`
    return head + (qtxt ? `<div style="${DIM};font-size:10px;margin-bottom:2px">${qtxt}</div>` : '') + list
  }

  function render() {
    const { base } = resolveConn()
    let snap = liveSnap
    let mode = 'live'
    if (!snap && !base) {
      // 回放兜底：地址未绑定/解析不出时渲染执行快照
      const raw = cur.outputs && cur.outputs.snapshot
      if (raw) {
        try { snap = typeof raw === 'string' ? JSON.parse(raw) : raw } catch { /* 坏快照 */ }
        if (snap) mode = 'replay'
      }
    }

    // 状态点 + 时间戳
    if (lastErr) dot.style.background = '#fbbf24'
    else if (snap) dot.style.background = mode === 'live' ? '#34d399' : '#94a3b8'
    else dot.style.background = '#64748b'
    timeLabel.textContent = snap
      ? (mode === 'replay' ? '执行快照 ' : '') + fmtTime(snap.ts)
      : ''

    // 提示条
    if (lastErr) {
      notice.style.display = 'block'
      notice.style.background = 'rgba(251,191,36,0.12)'
      notice.style.color = '#fbbf24'
      notice.textContent = '轮询失败：' + lastErr + (snap ? '（显示最后一次成功数据）' : '')
    } else if (!base && !snap) {
      notice.style.display = 'block'
      notice.style.background = 'rgba(148,163,184,0.12)'
      notice.style.color = 'rgba(255,255,255,0.5)'
      notice.textContent = '未解析到服务地址：请为 service_url 绑定流水线参数或填字面量'
    } else if (mode === 'replay') {
      notice.style.display = 'block'
      notice.style.background = 'rgba(148,163,184,0.12)'
      notice.style.color = 'rgba(255,255,255,0.5)'
      notice.textContent = '地址未绑定，显示节点运行时留存的快照'
    } else {
      notice.style.display = 'none'
    }

    if (!snap) {
      body.innerHTML = `<div style="${DIM};text-align:center;padding:20px 0">` +
        (base ? '连接中…' : '暂无数据') + '</div>'
      return
    }
    body.innerHTML =
      renderGpu(snap) + renderVram(snap) + renderModels(snap) +
      renderLeases(snap) + renderJobs(snap)
  }

  render()
  timer = setInterval(poll, POLL_MS)
  poll()

  return {
    update(next) {
      cur = next || {}
      render()
      // 参数可能刚改：立即重拉一轮
      poll()
    },
    unmount() {
      if (timer) clearInterval(timer)
      timer = null
      el.textContent = ''
    },
  }
}
