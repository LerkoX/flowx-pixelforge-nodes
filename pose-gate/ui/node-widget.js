/* pose-gate 节点 widget：骨架人工确认/修正编辑器（契约 apiVersion 1）。
 *
 * 节点卡片内只放状态行 + 小预览 +「打开骨架编辑器」按钮；点击弹出全屏
 * 浮层（position:fixed，与画布手势完全隔离），在浮层里拖拽关键点修正骨架，
 * 「确认放行」后经节点级 service-proxy 提交决策，服务端重渲染骨架继续流水线。
 * 设计期/回放态：只读展示。
 */
(function () {
  var BODY_LIMBS = [ // 1-based openpose limbSeq（同 controlnet_aux draw_bodypose）
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
    [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
    [1, 16], [16, 18], [3, 17], [6, 18]
  ]
  var BODY_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]
  ]
  var BODY_NAMES = ['鼻', '颈', '右肩', '右肘', '右腕', '左肩', '左肘', '左腕',
    '右髋', '右膝', '右踝', '左髋', '左膝', '左踝', '右眼', '左眼', '右耳', '左耳']
  var HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9],
    [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
  ]

  function rgb(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + (a == null ? 1 : a) + ')' }

  function mount(el, props) {
    var cur = props || {}
    var session = null
    var kp = null
    var origSubset = null
    var selected = -1
    var hoverIdx = -1
    var dragging = false
    var dragLast = null
    var chainDrag = true   // 连带子关节（拖肘带腕、拖膝带踝，保持肢体比例）
    var decided = false
    var decidedInfo = ''
    var showHand = true, showFace = false, showBg = true
    var bgImg = null, bgTried = false
    var timer = null
    var disposed = false
    var overlay = null
    var canvas = null, ctx = null
    var hintEl = null
    var activeGateKey = null   // 实际使用的闸门 key（可能与选中执行不同，靠 /ext/gates 发现）

    el.style.cssText = 'display:flex;flex-direction:column;gap:6px;padding:6px;font:12px/1.5 sans-serif;color:#cdd6e4;height:100%;box-sizing:border-box;overflow:hidden'

    var status = document.createElement('div')
    status.style.cssText = 'font-size:12px;color:#8b98ad;white-space:pre-line'
    el.appendChild(status)

    var thumb = document.createElement('img')
    thumb.style.cssText = 'display:none;max-width:100%;border-radius:6px;background:#000;align-self:center'
    el.appendChild(thumb)

    var openBtn = document.createElement('button')
    openBtn.textContent = '🦴 打开骨架编辑器'
    openBtn.style.cssText = 'display:none;padding:8px 12px;border:0;border-radius:6px;font-size:13px;cursor:pointer;color:#fff;background:#7c5cff;font-weight:600'
    openBtn.onclick = openEditor
    el.appendChild(openBtn)

    function proxyUrl(path, method) {
      return '/api/v1/executions/' + cur.execution.id + '/nodes/' +
        encodeURIComponent(cur.nodeId) + '/service-proxy?path=' +
        encodeURIComponent(path) + '&method=' + (method || 'GET')
    }

    function gateKey() {
      // 优先用发现的活跃闸门（UI 选中执行可能落后于实际运行执行）；
      // 未发现时回退到当前选中执行推导
      return activeGateKey || (cur.execution.id + ':' + cur.nodeId)
    }

    function discoverGate() {
      // 经当前执行的 service-proxy（仅需连通性）列出活跃闸门，
      // 按创建时间取本节点最新一个——解决 UI 选中执行与实际运行执行错位
      return fetch(proxyUrl('/ext/gates', 'GET'))
        .then(function (r) { if (!r.ok) throw new Error('' + r.status); return r.json() })
        .then(function (d) {
          var suffix = ':' + cur.nodeId
          var mine = (d.gates || []).filter(function (g) {
            return g.key.slice(-suffix.length) === suffix
          }).sort(function (a, b) { return b.created - a.created })
          if (mine.length) activeGateKey = mine[0].key
          return !!mine.length
        })
    }

    function poll() {
      if (disposed || decided) return
      if (!cur.execution) return
      fetch(proxyUrl('/ext/gate/' + encodeURIComponent(gateKey()), 'GET'))
        .then(function (r) { if (!r.ok) throw new Error('' + r.status); return r.json() })
        .then(function (s) {
          if (s.decision) { decided = true; decidedInfo = s.decision.action; renderState(); return }
          if (!session) {
            session = s
            kp = JSON.parse(JSON.stringify(s.keypoints))
            origSubset = kp.subset.slice()
            renderState()
          }
        })
        .catch(function () {
          // 当前 key 无会话：尝试重新发现活跃闸门（执行错位/页面刷新恢复）
          if (!session) discoverGate().catch(function () {})
        })
    }

    function submitDecision(action) {
      if (!kp || decided) return
      var body = action === 'render'
        ? { action: 'render', keypoints: kp, include_hand: showHand, include_face: showFace }
        : { action: 'confirm' }
      if (hintEl) hintEl.textContent = '提交中…'
      fetch(proxyUrl('/ext/gate/' + encodeURIComponent(gateKey()) + '/decision', 'POST'),
        { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json() })
        .then(function () {
          decided = true; decidedInfo = action
          closeEditor()
          renderState()
        })
        .catch(function (e) { if (hintEl) hintEl.textContent = '提交失败：' + e.message })
    }

    function vis(i) {
      return kp.subset[i] > 0.3 && kp.candidate[i][0] >= 0 && kp.candidate[i][1] >= 0
    }

    // 连带拖动链：拖动父关节时子关节一起平移（保持骨骼长度比例）
    var CHAIN = {
      0: [14, 15, 16, 17],
      2: [3, 4], 3: [4],
      5: [6, 7], 6: [7],
      8: [9, 10, 18, 19, 20], 9: [10, 18, 19, 20],
      11: [12, 13, 21, 22, 23], 12: [13, 21, 22, 23]
    }

    function draw() {
      if (!kp || !canvas) return
      var dw = canvas.width, dh = canvas.height
      ctx.clearRect(0, 0, dw, dh)
      ctx.fillStyle = '#000'; ctx.fillRect(0, 0, dw, dh)
      if (showBg && bgImg) {
        ctx.globalAlpha = 0.35
        ctx.drawImage(bgImg, 0, 0, dw, dh)
        ctx.globalAlpha = 1
      }
      var px = function (i) { return [kp.candidate[i][0] * dw, kp.candidate[i][1] * dh] }
      var i, a, b, pa, pb, p
      // 悬停高亮：先画提示圈，让用户知道会抓到哪个点
      if (hoverIdx >= 0 && vis(hoverIdx) && hoverIdx !== selected) {
        p = px(hoverIdx)
        ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 2
        ctx.beginPath(); ctx.arc(p[0], p[1], 13, 0, 7); ctx.stroke()
      }
      ctx.lineWidth = 4
      for (var l = 0; l < BODY_LIMBS.length; l++) {
        a = BODY_LIMBS[l][0] - 1; b = BODY_LIMBS[l][1] - 1
        if (!vis(a) || !vis(b)) continue
        pa = px(a); pb = px(b)
        ctx.strokeStyle = rgb(BODY_COLORS[l])
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke()
      }
      for (i = 0; i < 18; i++) {
        if (!vis(i)) continue
        p = px(i)
        ctx.fillStyle = rgb(BODY_COLORS[i])
        ctx.beginPath(); ctx.arc(p[0], p[1], i === selected ? 12 : 8, 0, 7); ctx.fill()
        if (i === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.stroke() }
      }
      ctx.fillStyle = '#ffff66'
      for (i = 18; i < 24; i++) {
        if (!vis(i)) continue
        p = px(i)
        ctx.beginPath(); ctx.arc(p[0], p[1], i === selected ? 9 : 6, 0, 7); ctx.fill()
        if (i === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke() }
      }
      if (showHand) {
        for (var h = 0; h < 2; h++) {
          var off = h === 0 ? 92 : 113
          ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,0.75)'
          for (var e = 0; e < HAND_EDGES.length; e++) {
            a = off + HAND_EDGES[e][0]; b = off + HAND_EDGES[e][1]
            if (!vis(a) || !vis(b)) continue
            pa = px(a); pb = px(b)
            ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke()
          }
          ctx.fillStyle = '#ff4444'
          for (i = 0; i < 21; i++) {
            if (!vis(off + i)) continue
            p = px(off + i)
            ctx.beginPath(); ctx.arc(p[0], p[1], (off + i) === selected ? 9 : 5.5, 0, 7); ctx.fill()
            if ((off + i) === selected) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke() }
          }
        }
      }
      if (showFace) {
        ctx.fillStyle = 'rgba(255,255,255,0.85)'
        for (i = 24; i < 92; i++) {
          if (!vis(i)) continue
          p = px(i)
          ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, 7); ctx.fill()
        }
      }
    }

    function hitTest(x, y) {
      if (!kp) return -1
      var dw = canvas.width, dh = canvas.height
      var best = -1, bestD = Math.max(28, dw * 0.06)  // 触屏友好的大命中半径
      var check = function (i) {
        if (!vis(i)) return
        var d = Math.hypot(kp.candidate[i][0] * dw - x, kp.candidate[i][1] * dh - y)
        if (d < bestD) { bestD = d; best = i }
      }
      for (var i = 0; i < 24; i++) check(i)
      if (showHand) for (i = 92; i < 134; i++) check(i)
      if (showFace) for (i = 24; i < 92; i++) check(i)
      return best
    }

    function labelOf(i) {
      if (i < 18) return '已选：' + BODY_NAMES[i]
      if (i < 24) return '已选：足部点 ' + (i - 18)
      if (i < 92) return '已选：面部点 ' + (i - 24)
      if (i < 113) return '已选：左手点 ' + (i - 92)
      return '已选：右手点 ' + (i - 113)
    }

    function mkBtn(text, bg) {
      var b = document.createElement('button')
      b.textContent = text
      b.style.cssText = 'padding:10px 16px;border:0;border-radius:8px;font-size:14px;cursor:pointer;color:#fff;background:' + bg + ';font-weight:600'
      return b
    }
    function mkChk(label, checked, onchange) {
      var l = document.createElement('label')
      l.style.cssText = 'font-size:13px;display:flex;align-items:center;gap:4px;cursor:pointer;color:#cdd6e4'
      var c = document.createElement('input')
      c.type = 'checkbox'; c.checked = checked
      c.onchange = function () { onchange(c.checked) }
      l.appendChild(c); l.appendChild(document.createTextNode(label))
      return l
    }

    function openEditor() {
      if (!kp || overlay) return
      // 底图（源图）懒加载
      if (!bgTried && session && session.source_id) {
        bgTried = true
        bgImg = new Image()
        bgImg.onload = function () { draw() }
        bgImg.onerror = function () { bgImg = null }
        bgImg.src = proxyUrl('/images/' + session.source_id, 'GET')
      }
      overlay = document.createElement('div')
      overlay.style.cssText = 'position:fixed;inset:0;z-index:99999;background:rgba(8,10,16,0.96);' +
        'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:12px;box-sizing:border-box'
      // 阻止事件穿透到画布
      ;['pointerdown', 'pointermove', 'pointerup', 'wheel', 'touchstart', 'touchmove', 'click'].forEach(function (ev) {
        overlay.addEventListener(ev, function (e) { e.stopPropagation() }, { passive: false })
      })

      var title = document.createElement('div')
      title.textContent = '🦴 骨架修正：拖动关键点到正确位置'
      title.style.cssText = 'color:#fff;font-size:15px;font-weight:600'
      overlay.appendChild(title)

      canvas = document.createElement('canvas')
      var maxW = Math.min(window.innerWidth * 0.92, 560)
      var maxH = window.innerHeight * 0.62
      var scale = Math.min(maxW / kp.width, maxH / kp.height)
      canvas.width = Math.round(kp.width * scale)
      canvas.height = Math.round(kp.height * scale)
      canvas.style.cssText = 'background:#000;border-radius:8px;touch-action:none;max-width:92vw'
      ctx = canvas.getContext('2d')
      overlay.appendChild(canvas)

      hintEl = document.createElement('div')
      hintEl.style.cssText = 'color:#8b98ad;font-size:13px;min-height:18px;text-align:center'
      hintEl.textContent = '点住彩色圆点拖动；「手/脸」控制是否参与绘制'
      overlay.appendChild(hintEl)

      var bar = document.createElement('div')
      bar.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:center'
      var btnOk = mkBtn('✅ 修正完成·放行', '#2d8a4e')
      var btnPass = mkBtn('⏭ 原样继续', '#55617a')
      var btnHide = mkBtn('🚫 隐藏选中点', '#8a5a2d')
      var btnRestore = mkBtn('↩ 恢复隐藏', '#55617a')
      var btnClose = mkBtn('✕ 关闭', '#3a4152')
      btnOk.onclick = function () { submitDecision('render') }
      btnPass.onclick = function () { submitDecision('confirm') }
      btnHide.onclick = function () {
        if (kp && selected >= 0) { kp.subset[selected] = 0; selected = -1; hintEl.textContent = '已隐藏该点'; draw() }
      }
      btnRestore.onclick = function () {
        if (kp && origSubset) { kp.subset = origSubset.slice(); selected = -1; hintEl.textContent = '已恢复全部隐藏点'; draw() }
      }
      btnClose.onclick = closeEditor
      bar.appendChild(btnOk); bar.appendChild(btnPass); bar.appendChild(btnHide)
      bar.appendChild(btnRestore)
      bar.appendChild(mkChk('手', showHand, function (v) { showHand = v; draw() }))
      bar.appendChild(mkChk('脸', showFace, function (v) { showFace = v; draw() }))
      bar.appendChild(mkChk('底图', showBg, function (v) { showBg = v; draw() }))
      bar.appendChild(mkChk('链动', chainDrag, function (v) { chainDrag = v }))
      bar.appendChild(btnClose)
      overlay.appendChild(bar)

      canvas.addEventListener('pointerdown', function (e) {
        if (decided || !kp) return
        var r = canvas.getBoundingClientRect()
        var i = hitTest(e.clientX - r.left, e.clientY - r.top)
        if (i >= 0) {
          selected = i; dragging = true
          dragLast = [e.clientX - r.left, e.clientY - r.top]
          try { canvas.setPointerCapture(e.pointerId) } catch (_) {}
          hintEl.textContent = labelOf(i) + '（拖到正确位置；误检点可「隐藏选中点」）'
          draw()
        }
      })
      canvas.addEventListener('pointermove', function (e) {
        if (!kp) return
        var r = canvas.getBoundingClientRect()
        var x = Math.min(Math.max(e.clientX - r.left, 0), r.width)
        var y = Math.min(Math.max(e.clientY - r.top, 0), r.height)
        if (dragging && selected >= 0) {
          // 位移量应用到目标点（连带模式下子关节一起平移，保持骨骼长度）
          var dx = (x - dragLast[0]) / r.width, dy = (y - dragLast[1]) / r.height
          dragLast = [x, y]
          var moves = [selected]
          if (chainDrag && CHAIN[selected]) moves = moves.concat(CHAIN[selected])
          for (var m = 0; m < moves.length; m++) {
            var j = moves[m]
            if (kp.subset[j] > 0 || j === selected) {
              kp.candidate[j] = [kp.candidate[j][0] + dx, kp.candidate[j][1] + dy]
            }
          }
          draw()
        } else if (!decided) {
          // 悬停高亮：提示当前会抓到哪个点
          var h = hitTest(x, y)
          if (h !== hoverIdx) {
            hoverIdx = h
            if (h >= 0) hintEl.textContent = labelOf(h)
            draw()
          }
        }
      })
      canvas.addEventListener('pointerup', function () { dragging = false })
      canvas.addEventListener('pointercancel', function () { dragging = false })

      document.body.appendChild(overlay)
      draw()
    }

    function closeEditor() {
      if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay)
      overlay = null; canvas = null; ctx = null; hintEl = null
    }

    function renderState() {
      if (decided) {
        openBtn.style.display = 'none'
        status.textContent = '✅ 已放行（' + (decidedInfo === 'render' ? '修正骨架' : '原骨架') + '），流水线继续'
        return
      }
      if (session && !decided) {
        // 闸门会话存活且未决策 = 节点正在等待（以服务端会话为准，不看 UI 选中执行）
        openBtn.style.display = 'block'
        status.textContent = '⏸ 等待人工确认骨架'
        if (session.image_id && !thumb.src) {
          thumb.src = proxyUrl('/images/' + session.image_id + '?thumb=300', 'GET')
          thumb.style.display = 'block'
        }
        return
      }
      openBtn.style.display = 'none'
      if (cur.status === 'running') {
        status.textContent = '⏳ 姿态检测中…（DWPose CPU 推理约 10~30s）'
        return
      }
      if (cur.outputs && cur.outputs.image) {
        status.textContent = '完成（gate_action=' + (cur.outputs.gate_action || '-') + '）'
        return
      }
      status.textContent = '执行到此节点时，点「打开骨架编辑器」确认/修正姿态'
    }

    timer = setInterval(poll, 3000)
    poll()

    renderState()
    return {
      update: function (p) {
        var wasRunning = cur.status
        cur = p || {}
        if (cur.outputs && cur.outputs.gate_action && !decided) {
          decided = true
          decidedInfo = cur.outputs.gate_action === 'render' ? 'render' : 'confirm'
          closeEditor()
        }
        if (cur.status !== wasRunning) poll()
        renderState()
      },
      unmount: function () { disposed = true; if (timer) clearInterval(timer); closeEditor() }
    }
  }

  if (window.FlowXNodeWidget && window.FlowXNodeWidget.define) {
    window.FlowXNodeWidget.define(mount)
  }
  if (typeof module !== 'undefined') module.exports = mount
})()
